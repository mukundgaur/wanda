import time
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT
from .data import get_loaders

from .ablate import AblateGPT

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0
    
def prune_wanda_block(args, model, tokenizer, device=torch.device("cuda:0"), block_size=16):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            rows, cols = W_metric.shape
            pad_rows = (block_size - rows % block_size) % block_size
            pad_cols = (block_size - cols % block_size) % block_size
            W_metric_padded = torch.nn.functional.pad(W_metric, (0, pad_cols, 0, pad_rows), value=0)

            br = W_metric_padded.shape[0] // block_size
            bc = W_metric_padded.shape[1] // block_size

            # Compute mean Wanda score per block
            block_scores = (
                W_metric_padded
                .reshape(br, block_size, bc, block_size)
                .permute(0, 2, 1, 3)
                .reshape(br * bc, -1)
                .mean(dim=1)
                .reshape(br, bc)
            )

            # Prune lowest-scoring blocks globally
            num_blocks = br * bc
            num_prune = int(num_blocks * args.sparsity_ratio)
            if num_prune > 0:
                threshold = torch.topk(block_scores.flatten(), num_prune, largest=False)[0][-1]
                _, prune_indices = torch.topk(block_scores.flatten(), num_prune, largest=False)
                block_mask = torch.zeros(num_blocks, dtype=torch.bool, device=W_metric.device)
                block_mask.scatter_(0, prune_indices, True)
                block_mask = block_mask.reshape(br, bc)
            else:
                block_mask = torch.zeros(br, bc, dtype=torch.bool, device=W_metric.device)

            # Expand block mask to full weight dimensions
            W_mask_padded = (
                block_mask
                .unsqueeze(2).unsqueeze(3)
                .expand(br, bc, block_size, block_size)
                .permute(0, 2, 1, 3)
                .reshape(W_metric_padded.shape)
            )
            W_mask = W_mask_padded[:rows, :cols]

            subset[name].weight.data[W_mask] = 0

        # Update inputs for next layer
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibration data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Shared helpers for CUT-BELL structured pruning strategies
# ---------------------------------------------------------------------------

def _block_view(W: torch.Tensor, B: int):
    """Reshape (rows, cols) → (nb_r, nb_c, B, B), padding to B multiples."""
    rows, cols = W.shape
    pr = (-rows) % B
    pc = (-cols) % B
    W_pad = F.pad(W, (0, pc, 0, pr), value=0)
    nb_r = W_pad.shape[0] // B
    nb_c = W_pad.shape[1] // B
    return W_pad.reshape(nb_r, B, nb_c, B).permute(0, 2, 1, 3), (rows, cols)


def _from_block_view(blocks: torch.Tensor, orig, B: int) -> torch.Tensor:
    """Inverse of _block_view: (nb_r, nb_c, B, B) → cropped (rows, cols)."""
    nb_r, nb_c = blocks.shape[:2]
    return blocks.permute(0, 2, 1, 3).reshape(nb_r * B, nb_c * B)[:orig[0], :orig[1]]


def _block_norms(blocks: torch.Tensor) -> torch.Tensor:
    """L2 norm of each block: (nb_r, nb_c, B, B) → (nb_r, nb_c)."""
    return blocks.pow(2).sum(dim=(2, 3)).sqrt()


def _expand_block_mask(block_mask: torch.Tensor, orig, B: int) -> torch.Tensor:
    """Expand boolean block mask (nb_r, nb_c) → element mask cropped to orig."""
    nb_r, nb_c = block_mask.shape
    return (
        block_mask.unsqueeze(2).unsqueeze(3)
        .expand(nb_r, nb_c, B, B)
        .permute(0, 2, 1, 3)
        .reshape(nb_r * B, nb_c * B)
    )[:orig[0], :orig[1]]


def _top_norm_mask(col_imp: torch.Tensor, n_keep: int) -> torch.Tensor:
    """Boolean mask of shape (win_size,): True where block-column is kept."""
    win_size = col_imp.shape[0]
    if n_keep >= win_size:
        return torch.ones(win_size, dtype=torch.bool, device=col_imp.device)
    _, top_idx = torch.topk(col_imp, n_keep, largest=True)
    mask = torch.zeros(win_size, dtype=torch.bool, device=col_imp.device)
    mask[top_idx] = True
    return mask


# ---------------------------------------------------------------------------
# Strategy 1: Block-magnitude
# Prune lowest-L2-norm B×B blocks globally.
# ---------------------------------------------------------------------------

def prune_block_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), block_size=16):
    layers = model.model.layers
    B = block_size

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.float()
            blocks, orig = _block_view(W, B)
            nb_r, nb_c = blocks.shape[:2]
            norms = _block_norms(blocks)          # (nb_r, nb_c)
            total_blocks = nb_r * nb_c
            n_prune = int(total_blocks * args.sparsity_ratio)
            if n_prune > 0:
                thresh = torch.kthvalue(norms.flatten(), n_prune + 1).values
                block_mask = (norms < thresh).reshape(nb_r, nb_c)
            else:
                block_mask = torch.zeros(nb_r, nb_c, dtype=torch.bool, device=W.device)
            W_mask = _expand_block_mask(block_mask, orig, B)
            subset[name].weight.data[W_mask] = 0


# ---------------------------------------------------------------------------
# Strategy 2: Column-block
# Prune entire B-wide block-columns globally by ascending summed norm.
# ---------------------------------------------------------------------------

def prune_column_block(args, model, tokenizer, device=torch.device("cuda:0"), block_size=16):
    layers = model.model.layers
    B = block_size

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.float()
            blocks, orig = _block_view(W, B)
            nb_r, nb_c = blocks.shape[:2]
            col_imp = _block_norms(blocks).sum(dim=0)   # (nb_c,)
            n_prune_cols = int(nb_c * args.sparsity_ratio)
            if n_prune_cols > 0:
                thresh = torch.kthvalue(col_imp, n_prune_cols + 1).values
                block_mask = (col_imp < thresh).unsqueeze(0).expand(nb_r, -1).contiguous()
            else:
                block_mask = torch.zeros(nb_r, nb_c, dtype=torch.bool, device=W.device)
            W_mask = _expand_block_mask(block_mask, orig, B)
            subset[name].weight.data[W_mask] = 0


# ---------------------------------------------------------------------------
# Strategy 3: Batch-aligned
# Per-batch keep top block-columns by combined norm; different batches can
# retain different columns.
# ---------------------------------------------------------------------------

def prune_batch_aligned(args, model, tokenizer, device=torch.device("cuda:0"),
                        block_size=16, channels=16):
    layers = model.model.layers
    B = block_size
    density = 1.0 - args.sparsity_ratio

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.float()
            blocks, orig = _block_view(W, B)
            nb_r, nb_c = blocks.shape[:2]
            norms = _block_norms(blocks)                 # (nb_r, nb_c)
            n_keep = max(1, round(density * nb_c))
            block_mask = torch.zeros(nb_r, nb_c, dtype=torch.bool, device=W.device)
            for start in range(0, nb_r, channels):
                end = min(start + channels, nb_r)
                col_imp = norms[start:end].sum(dim=0)    # (nb_c,)
                keep = _top_norm_mask(col_imp, min(n_keep, nb_c))
                block_mask[start:end] = ~keep.unsqueeze(0)
            W_mask = _expand_block_mask(block_mask, orig, B)
            subset[name].weight.data[W_mask] = 0


# ---------------------------------------------------------------------------
# Strategy 4: Shared-block n:m
# Within each window of m_b block-columns, keep exactly n_b per batch.
# ---------------------------------------------------------------------------

def prune_shared_block_nm(args, model, tokenizer, device=torch.device("cuda:0"),
                          block_size=16, channels=16, m_b=4):
    layers = model.model.layers
    B = block_size
    density = 1.0 - args.sparsity_ratio
    n_b = max(1, min(m_b, round(density * m_b)))

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            if i == 0 and "gate_proj" in name:
                np.save("gate_proj_weights.npy", subset[name].weight.data.float().cpu().numpy())
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.float()
            blocks, orig = _block_view(W, B)
            nb_r, nb_c = blocks.shape[:2]
            norms = _block_norms(blocks)                 # (nb_r, nb_c)
            block_mask = torch.zeros(nb_r, nb_c, dtype=torch.bool, device=W.device)
            for win_start in range(0, nb_c, m_b):
                win_end = min(win_start + m_b, nb_c)
                n_keep = min(n_b, win_end - win_start)
                for batch_start in range(0, nb_r, channels):
                    batch_end = min(batch_start + channels, nb_r)
                    col_imp = norms[batch_start:batch_end, win_start:win_end].sum(dim=0)
                    keep = _top_norm_mask(col_imp, n_keep)
                    block_mask[batch_start:batch_end, win_start:win_end] = ~keep.unsqueeze(0)
            W_mask = _expand_block_mask(block_mask, orig, B)
            subset[name].weight.data[W_mask] = 0
            if i == 0 and "gate_proj" in name:
                mask_torch = (subset[name].weight.data == 0).cpu().numpy()
                np.save("mask_torch.npy", mask_torch)


# ---------------------------------------------------------------------------
# Strategy 5: Perm-shared-block n:m (Strategy B)
# Sort block-columns within each window by global norm, keep first n_b
# positions for ALL rows, then restore original column order.
# ---------------------------------------------------------------------------

def prune_perm_shared_block_nm(args, model, tokenizer, device=torch.device("cuda:0"),
                               block_size=16, channels=16, m_b=10):
    # channels unused: kept for a uniform call signature across strategies
    layers = model.model.layers
    B = block_size
    density = 1.0 - args.sparsity_ratio
    n_b = max(1, min(m_b, round(density * m_b)))

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            if i == 0 and "gate_proj" in name:
                np.save("gate_proj_weights.npy", subset[name].weight.data.float().cpu().numpy())
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.float()
            blocks, orig = _block_view(W, B)
            _, nb_c = blocks.shape[:2]
            bc_imp = _block_norms(blocks).sum(dim=0)     # (nb_c,) global importance

            # Build within-window descending-sort column permutation
            perm = torch.empty(nb_c, dtype=torch.long, device=W.device)
            for win_start in range(0, nb_c, m_b):
                win_end = min(win_start + m_b, nb_c)
                window = torch.arange(win_start, win_end, device=W.device)
                order = torch.argsort(bc_imp[window], descending=True)
                perm[win_start:win_end] = window[order]

            # Permute columns, zero out globally least-important tail per window
            result = blocks[:, perm, :, :].clone()
            for win_start in range(0, nb_c, m_b):
                win_end = min(win_start + m_b, nb_c)
                n_keep = min(n_b, win_end - win_start)
                if n_keep < win_end - win_start:
                    result[:, win_start + n_keep:win_end, :, :] = 0.0

            # Restore original column order and write back
            inv_perm = torch.argsort(perm)
            result = result[:, inv_perm, :, :]
            W_pruned = _from_block_view(result, orig, B)
            subset[name].weight.data[:] = W_pruned.to(subset[name].weight.dtype)
            if i == 0 and "gate_proj" in name:
                mask_torch = (subset[name].weight.data == 0).cpu().numpy()
                np.save("mask_torch.npy", mask_torch)


# ---------------------------------------------------------------------------
# Strategy 6: Batch-row-perm n:m (Strategy F)
# Greedily assign block-rows to batches by column-overlap similarity, apply
# shared-block n:m on permuted rows, then restore original row order.
# ---------------------------------------------------------------------------

def prune_batch_row_perm_nm(args, model, tokenizer, device=torch.device("cuda:0"),
                            block_size=16, channels=16, m_b=10):
    layers = model.model.layers
    B = block_size
    density = 1.0 - args.sparsity_ratio
    n_b = max(1, min(m_b, round(density * m_b)))

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            if i == 0 and "gate_proj" in name:
                np.save("gate_proj_weights_2.npy", subset[name].weight.data.float().cpu().numpy())
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.float()
            blocks, orig = _block_view(W, B)
            nb_r, nb_c = blocks.shape[:2]
            norms = _block_norms(blocks)             # (nb_r, nb_c)

            # Build top-k mask per block-row for similarity (numpy for greedy loop)
            n_top = max(1, round(density * nb_c))
            norms_np = norms.cpu().numpy()
            top_mask = np.zeros((nb_r, nb_c), dtype=np.float32)
            if n_top < nb_c:
                top_idx = np.argpartition(norms_np, nb_c - n_top, axis=1)[:, nb_c - n_top:]
            else:
                top_idx = np.tile(np.arange(nb_c), (nb_r, 1))
            for r in range(nb_r):
                top_mask[r, top_idx[r]] = 1.0

            # Pairwise row similarity via matmul, greedy batch assignment
            sim = top_mask @ top_mask.T              # (nb_r, nb_r)
            np.fill_diagonal(sim, -1.0)
            assigned = np.zeros(nb_r, dtype=bool)
            batch_order = []
            while not assigned.all():
                free = np.where(~assigned)[0]
                if len(free) == 1:
                    batch_order.append(int(free[0]))
                    assigned[free[0]] = True
                    continue
                sub_sim = sim[np.ix_(free, free)]
                best_flat = int(sub_sim.argmax())
                seed = int(free[best_flat // len(free)])
                assigned[seed] = True
                batch = [seed]
                batch_union = top_mask[seed].copy()
                while len(batch) < channels:
                    free_now = np.where(~assigned)[0]
                    if len(free_now) == 0:
                        break
                    cand_sim = top_mask[free_now] @ batch_union
                    best = int(free_now[int(cand_sim.argmax())])
                    assigned[best] = True
                    batch.append(best)
                    np.maximum(batch_union, top_mask[best], out=batch_union)
                batch_order.extend(batch)

            perm = torch.tensor(batch_order, dtype=torch.long, device=W.device)
            blocks_p = blocks[perm, :, :, :]
            norms_p = _block_norms(blocks_p)

            # Shared-block n:m on permuted rows
            result = blocks_p.clone()
            for win_start in range(0, nb_c, m_b):
                win_end = min(win_start + m_b, nb_c)
                n_keep = min(n_b, win_end - win_start)
                for batch_start in range(0, nb_r, channels):
                    batch_end = min(batch_start + channels, nb_r)
                    col_imp = norms_p[batch_start:batch_end, win_start:win_end].sum(dim=0)
                    keep = _top_norm_mask(col_imp, n_keep)
                    result[batch_start:batch_end, win_start:win_end] *= (
                        keep.float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    )

            # Restore original row order and write back
            inv_perm = torch.argsort(perm)
            result = result[inv_perm, :, :, :]
            W_pruned = _from_block_view(result, orig, B)
            subset[name].weight.data[:] = W_pruned.to(subset[name].weight.dtype)
            if i == 0 and "gate_proj" in name:
                mask_torch = (subset[name].weight.data == 0).cpu().numpy()
                np.save("mask_torch_2.npy", mask_torch)


# ---------------------------------------------------------------------------
# Element-level helper: per-row N:M in column windows
# Used by all scalar / hierarchical strategies below.
# ---------------------------------------------------------------------------

def _row_nm_in_windows(W: torch.Tensor, n_e: int, m_e: int) -> torch.Tensor:
    """Apply per-row N:M element sparsity in windows of m_e columns.

    For each row and each window of m_e consecutive columns, keep the n_e
    elements with the largest absolute value; zero the rest.

    Parameters
    ----------
    W   : (rows, cols) float tensor
    n_e : elements to keep per window
    m_e : window size in columns

    Returns
    -------
    result : same shape as W, with at most n_e non-zeros per row-window
    """
    rows, cols = W.shape
    result = torch.zeros_like(W)
    for w in range(0, cols, m_e):
        seg = W[:, w:w + m_e]
        seg_len = seg.shape[1]
        n_k = min(n_e, seg_len)
        if n_k >= seg_len:
            result[:, w:w + m_e] = seg
        else:
            # Match numpy: threshold = n_k-th largest = (seg_len - n_k + 1)-th smallest.
            # Keep all abs >= thresh so ties at the boundary are kept, not broken arbitrarily.
            thresh = torch.kthvalue(seg.abs(), seg_len - n_k + 1, dim=1).values  # (rows,)
            result[:, w:w + m_e] = seg * (seg.abs() >= thresh.unsqueeze(1))
    return result


# ---------------------------------------------------------------------------
# Strategy 7: Scalar magnitude
# Global element-level magnitude pruning — no block structure.
# ---------------------------------------------------------------------------

def prune_scalar_magnitude(args, model, tokenizer, device=torch.device("cuda:0")):
    """Global element-level magnitude pruning (no block structure).

    Keeps the top (1 - sparsity_ratio) × total elements by |W| with a single
    global threshold.  Best perplexity at a given element budget; poor col_union
    because each row independently picks its best elements, so almost every
    block-column stays active in every CUT-BELL batch.
    """
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            if i == 0 and "gate_proj" in name:
                np.save("gate_proj_weights.npy", subset[name].weight.data.float().cpu().numpy())
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.float()
            n_prune = int(W.numel() * args.sparsity_ratio)
            if n_prune > 0:
                # Match numpy: threshold = (n_prune+1)-th smallest (np.partition index n_prune),
                # keep all >= thresh so ties at boundary are kept, not pruned.
                thresh = torch.kthvalue(W.abs().flatten(), n_prune + 1).values
                W_mask = W.abs() < thresh
            else:
                W_mask = torch.zeros_like(W, dtype=torch.bool)
            subset[name].weight.data[W_mask] = 0
            if i == 0 and "gate_proj" in name:
                mask_torch = (subset[name].weight.data == 0).cpu().numpy()
                np.save("mask_torch.npy", mask_torch)


# ---------------------------------------------------------------------------
# Strategy 8: Scalar column-sorted N:M
# Sort columns by descending global L2 norm, apply per-row element N:M in
# col-importance windows, undo permutation.
# ---------------------------------------------------------------------------

def prune_scalar_col_sorted_nm(args, model, tokenizer, device=torch.device("cuda:0"),
                                m_e=32):
    """Global column-sort (by L2 norm) + per-row element N:M in col-importance windows.

    Columns are sorted by descending global L2 norm; per-row N:M element sparsity
    keeps top-n_e elements in each window of m_e columns; the column permutation
    is then undone.  High-importance columns concentrate kept elements in the
    leading half of each window, giving col_union ≈ density × nb_c (similar to
    column_block) while selecting individual elements for better perplexity.

    Parameters
    ----------
    m_e : window size in columns (default 32 = 2 × block_size).
          Use m_e = 2 × block_size or larger to achieve column_block-like col_union.
    """
    layers = model.model.layers
    density = 1.0 - args.sparsity_ratio
    n_e = max(1, round(density * m_e))

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            if i == 0 and "gate_proj" in name:
                np.save("gate_proj_weights.npy", subset[name].weight.data.float().cpu().numpy())
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.float()
            # Compute permutation via numpy to be bit-exact with the reference:
            # np.linalg.norm stays in float32 and np.argsort(-norms) must match.
            W_np = W.cpu().numpy()
            col_norms_np = np.linalg.norm(W_np, axis=0)          # float32, same as ref
            perm_np = np.argsort(-col_norms_np)                   # descending, same as ref
            perm = torch.from_numpy(perm_np).to(device=W.device, dtype=torch.long)
            inv_perm = torch.argsort(perm, stable=True)
            W_perm = W[:, perm]
            W_pruned = _row_nm_in_windows(W_perm, n_e, m_e)
            subset[name].weight.data[:] = W_pruned[:, inv_perm].to(
                subset[name].weight.dtype)
            if i == 0 and "gate_proj" in name:
                mask_torch = (subset[name].weight.data == 0).cpu().numpy()
                np.save("mask_torch.npy", mask_torch)


# ---------------------------------------------------------------------------
# Strategy 9: Scalar batch-sorted N:M
# Per-CUT-BELL-batch column-sort + per-row element N:M — scalar analogue of
# batch_aligned.
# ---------------------------------------------------------------------------

def prune_scalar_batch_sorted_nm(args, model, tokenizer, device=torch.device("cuda:0"),
                                  block_size=16, channels=16, m_e=32):
    """Per-CUT-BELL-batch column-sort + per-row element N:M.

    Same as prune_scalar_col_sorted_nm but the column permutation is recomputed
    for each CUT-BELL batch of (channels × block_size) rows, so each batch
    focuses on the columns most important for its own rows.  Expected col_union
    ≈ batch_aligned (≈ density × nb_c per batch); perplexity better than
    batch_aligned because individual elements rather than whole blocks are selected.

    Parameters
    ----------
    block_size : block height / CUT-BELL row-granularity (default 16)
    channels   : number of block-rows per CUT-BELL batch (default 16)
    m_e        : element N:M window size in columns (default 32)
    """
    layers = model.model.layers
    density = 1.0 - args.sparsity_ratio
    n_e = max(1, round(density * m_e))
    B = block_size

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            if i == 0 and "gate_proj" in name:
                np.save("gate_proj_weights.npy", subset[name].weight.data.float().cpu().numpy())
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.float()
            rows, cols = W.shape
            nb_r = (rows + B - 1) // B
            result = torch.zeros_like(W)
            for batch_start in range(0, nb_r, channels):
                batch_end = min(batch_start + channels, nb_r)
                r0 = batch_start * B
                r1 = min(batch_end * B, rows)
                batch = W[r0:r1]                              # (batch_rows, cols)
                batch_np = batch.cpu().numpy()
                col_norms_np = np.linalg.norm(batch_np, axis=0)
                perm_np = np.argsort(-col_norms_np)
                perm = torch.from_numpy(perm_np).to(device=W.device, dtype=torch.long)
                inv_perm = torch.argsort(perm, stable=True)
                pruned = _row_nm_in_windows(batch[:, perm], n_e, m_e)
                result[r0:r1] = pruned[:, inv_perm]
            subset[name].weight.data[:] = result.to(subset[name].weight.dtype)
            if i == 0 and "gate_proj" in name:
                mask_torch = (subset[name].weight.data == 0).cpu().numpy()
                np.save("mask_torch.npy", mask_torch)


# ---------------------------------------------------------------------------
# Strategy 10: Hierarchical block + scalar N:M (batch-aligned base)
# Stage 1: batch-aligned block selection at an elevated block density.
# Stage 2: element N:M within each active block.
# ---------------------------------------------------------------------------

def prune_hierarchical_block_scalar_nm(args, model, tokenizer,
                                        device=torch.device("cuda:0"),
                                        block_size=16, channels=16,
                                        n_e=8, m_e=16):
    """Two-level hierarchical pruning: batch-aligned block selection + within-block N:M.

    Stage 1 — block selection
        block_density = target_density / (n_e / m_e).
        Apply batch-aligned block selection at block_density (col_union = batch_aligned
        at block_density, which is HIGHER than target_density → col_union is higher
        than plain batch_aligned at target_density).

    Stage 2 — within-block element N:M
        In each active block row, keep n_e out of every m_e elements by |W|.

    Total element density ≈ block_density × (n_e / m_e) = target_density.
    Better perplexity than plain batch_aligned at target_density because element-
    level selection discards low-magnitude weights within active blocks.

    Parameters
    ----------
    n_e, m_e : within-block N:M ratio (default 8:16 = 50% within each active block;
               m_e must equal block_size for standard within-block N:M)
    """
    layers = model.model.layers
    B = block_size
    density = 1.0 - args.sparsity_ratio
    within_density = n_e / m_e
    block_density = min(1.0, density / within_density)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            if i == 0 and "gate_proj" in name:
                np.save("gate_proj_weights.npy", subset[name].weight.data.float().cpu().numpy())
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.float()
            blocks, orig = _block_view(W, B)
            nb_r, nb_c = blocks.shape[:2]
            norms = _block_norms(blocks)                      # (nb_r, nb_c)
            n_keep = max(1, round(block_density * nb_c))

            # Stage 1: batch-aligned block selection
            block_mask = torch.zeros(nb_r, nb_c, dtype=torch.bool, device=W.device)
            for start in range(0, nb_r, channels):
                end = min(start + channels, nb_r)
                col_imp = norms[start:end].sum(dim=0)
                keep = _top_norm_mask(col_imp, min(n_keep, nb_c))
                block_mask[start:end] = keep.unsqueeze(0)

            result = blocks * block_mask.unsqueeze(2).unsqueeze(3).float()

            # Stage 2: element N:M within each active block
            for br in range(nb_r):
                for bc in range(nb_c):
                    if not block_mask[br, bc]:
                        continue
                    result[br, bc] = _row_nm_in_windows(result[br, bc], n_e, m_e)

            W_pruned = _from_block_view(result, orig, B)
            subset[name].weight.data[:] = W_pruned.to(subset[name].weight.dtype)
            if i == 0 and "gate_proj" in name:
                mask_torch = (subset[name].weight.data == 0).cpu().numpy()
                np.save("mask_torch.npy", mask_torch)


# ---------------------------------------------------------------------------
# Strategy 11: Fully hierarchical N:M (shared-block + element N:M)
# Stage 1: shared-block n:m block selection (optimal col_union).
# Stage 2: element N:M within each active block.
# ---------------------------------------------------------------------------

def prune_hierarchical_sbnm_scalar_nm(args, model, tokenizer,
                                       device=torch.device("cuda:0"),
                                       block_size=16, channels=16,
                                       n_b=1, m_b=2, n_e=8, m_e=16):
    """Fully hierarchical N:M: shared_block_nm at block level + element N:M within.

    Block level  : in each window of m_b block-columns, keep exactly n_b per
                   CUT-BELL batch (prune_shared_block_nm selection, col_union =
                   block_fill exactly).
    Element level: in each window of m_e elements within each active block row,
                   keep exactly n_e elements by |W|.

    Total element density = (n_b / m_b) × (n_e / m_e).
    col_union = shared_block_nm at (n_b:m_b) — the theoretical minimum.
    Most hardware-friendly variant: strict N:M at both levels, optimal col_union,
    fine-grained element selection for perplexity.

    Parameters
    ----------
    n_b, m_b : block-level N:M (default 1:2 = 50% block density)
    n_e, m_e : element-level N:M within each active block row (default 8:16 = 50%)
    """
    layers = model.model.layers
    B = block_size

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.float()
            blocks, orig = _block_view(W, B)
            nb_r, nb_c = blocks.shape[:2]
            norms = _block_norms(blocks)                      # (nb_r, nb_c)

            # Stage 1: shared-block n:m selection
            block_mask = torch.zeros(nb_r, nb_c, dtype=torch.bool, device=W.device)
            for win_start in range(0, nb_c, m_b):
                win_end = min(win_start + m_b, nb_c)
                n_keep = min(n_b, win_end - win_start)
                for batch_start in range(0, nb_r, channels):
                    batch_end = min(batch_start + channels, nb_r)
                    col_imp = norms[batch_start:batch_end, win_start:win_end].sum(dim=0)
                    keep = _top_norm_mask(col_imp, n_keep)
                    block_mask[batch_start:batch_end, win_start:win_end] = keep.unsqueeze(0)

            result = blocks * block_mask.unsqueeze(2).unsqueeze(3).float()

            # Stage 2: element N:M within each active block
            for br in range(nb_r):
                for bc in range(nb_c):
                    if not block_mask[br, bc]:
                        continue
                    result[br, bc] = _row_nm_in_windows(result[br, bc], n_e, m_e)

            W_pruned = _from_block_view(result, orig, B)
            subset[name].weight.data[:] = W_pruned.to(subset[name].weight.dtype)


# ---------------------------------------------------------------------------
# Activation-aware strategies
# Use Wanda importance = |W[i,j]| × sqrt(scaler_row[j]) for scoring.
# All functions below require calibration data (same boilerplate as prune_wanda).
# ---------------------------------------------------------------------------

def prune_wanda_element(args, model, tokenizer, device=torch.device("cuda:0")):
    """Global element-level Wanda pruning with a single global threshold.

    Keeps the top (1 - sparsity_ratio) × total elements by Wanda score
    |W[i,j]| × sqrt(scaler_row[j]) using one global threshold across the whole
    weight matrix.  Unlike prune_wanda (which prunes a uniform fraction per row),
    this allows rows with consistently high activation-weighted weights to retain
    more elements.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev), outs.to(dev),
                attention_mask.to(dev), position_ids.to(dev))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = (torch.abs(subset[name].weight.data)
                        * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))))
            n_prune = int(W_metric.numel() * args.sparsity_ratio)
            if n_prune > 0:
                thresh = torch.kthvalue(W_metric.flatten(), n_prune + 1).values
                W_mask = W_metric < thresh
            else:
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
            subset[name].weight.data[W_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_block_wanda_nm(args, model, tokenizer, device=torch.device("cuda:0"),
                          block_size=16, channels=16, m_b=4):
    """Shared-block n:m with Wanda importance for block-column selection.

    Same structure as prune_shared_block_nm but scores each block-column by the
    total Wanda importance (sum of |W[i,j]| × sqrt(scaler_row[j]) over all rows
    in the batch and all elements in the block) instead of L2 norm.  Activation-
    weighted column selection focuses pruning on input channels that matter least
    for the calibration distribution.

    Parameters
    ----------
    block_size : block height and width (default 16)
    channels   : CUT-BELL batch size in block-rows (default 16)
    m_b        : block-column window size for N:M (default 4)
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device)

    layers = model.model.layers
    B = block_size
    density = 1.0 - args.sparsity_ratio
    n_b = max(1, min(m_b, round(density * m_b)))

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev), outs.to(dev),
                attention_mask.to(dev), position_ids.to(dev))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data.float()
            channel_norms = torch.sqrt(wrapped_layers[name].scaler_row)  # (in_features,)
            W_imp = W.abs() * channel_norms.unsqueeze(0)                 # (out, in)

            blocks_W, orig = _block_view(W, B)
            blocks_I, _    = _block_view(W_imp, B)
            nb_r, nb_c     = blocks_W.shape[:2]
            result         = blocks_W.clone()

            for win_start in range(0, nb_c, m_b):
                win_end = min(win_start + m_b, nb_c)
                n_keep  = min(n_b, win_end - win_start)
                for batch_start in range(0, nb_r, channels):
                    batch_end = min(batch_start + channels, nb_r)
                    # Sum Wanda importance per block-column across batch + block elements
                    col_imp = blocks_I[batch_start:batch_end,
                                       win_start:win_end].sum(dim=(0, 2, 3))
                    keep = _top_norm_mask(col_imp, n_keep)
                    result[batch_start:batch_end, win_start:win_end] *= (
                        keep.float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

            W_pruned = _from_block_view(result, orig, B)
            subset[name].weight.data[:] = W_pruned.to(subset[name].weight.dtype)

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_scalar_block_wanda(args, model, tokenizer, device=torch.device("cuda:0"),
                              block_size=16, channels=16, m_b=4,
                              alpha=0.0, scalar_sparsity=None):
    """Scalar-aware block pruning with Wanda importance (full AiM CUT-BELL pipeline).

    Implements the complete scalar_block_prune pipeline from
    aim/benchmarks/scalar_block_pruning.py, driven by Wanda activation statistics:

      1. Wanda importance scores: |W[i,j]| × channel_norm[j]
      2. Scalar pruning: keep top scalar_sparsity elements globally by Wanda score
      3. Column permutation: sort block-columns by total Wanda score (descending)
      4. Row permutation: cluster block-rows with similar active-column patterns
      5. Greedy n:m block-column selection per CUT-BELL batch (enforces identical
         col_union to prune_shared_block_nm / prune_block_wanda_nm)
      6. Apply both masks; undo permutations to restore original weight layout

    The gain vs prune_block_wanda_nm comes from steps 2-4: the greedy selector
    chooses blocks whose surviving scalar elements carry the highest activation-
    weighted importance, and within-block "holes" reduce effective weight count
    without increasing col_union.

    Parameters
    ----------
    block_size      : block height and width (default 16)
    channels        : CUT-BELL batch size in block-rows (default 16)
    m_b             : block-column window size for n:m (default 4)
    alpha           : density exponent for block scoring (0 = pure importance,
                      >0 favours denser blocks)
    scalar_sparsity : fraction of elements to keep in scalar step;
                      defaults to (1 - args.sparsity_ratio) if None
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent.parent.parent / "aim" / "benchmarks"))
    from scalar_block_pruning import scalar_block_prune  # noqa: E402

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                seqlen=model.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device)

    layers = model.model.layers
    density = 1.0 - args.sparsity_ratio
    n_b = max(1, min(m_b, round(density * m_b)))
    ss = scalar_sparsity if scalar_sparsity is not None else density

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev), outs.to(dev),
                attention_mask.to(dev), position_ids.to(dev))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_np = subset[name].weight.data.float().cpu().numpy()
            # Represent activations as a column-norm proxy: (in_features, 1).
            # scalar_block_prune only needs ||X[j,:]||_2, which equals channel_norm[j].
            channel_norms = torch.sqrt(wrapped_layers[name].scaler_row).cpu().numpy()
            X_proxy = channel_norms[:, np.newaxis]   # (in_features, 1)

            W_pruned_np, _, _, _ = scalar_block_prune(
                W_np, X_proxy,
                n=n_b, m=m_b,
                channels=channels,
                alpha=alpha,
                scalar_sparsity=ss,
            )
            subset[name].weight.data[:] = torch.from_numpy(W_pruned_np).to(
                device=subset[name].weight.device,
                dtype=subset[name].weight.dtype,
            )

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                                position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()