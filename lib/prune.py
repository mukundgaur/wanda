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
                _, prune_idx = torch.topk(norms.flatten(), n_prune, largest=False)
                block_mask = torch.zeros(total_blocks, dtype=torch.bool, device=W.device)
                block_mask.scatter_(0, prune_idx, True)
                block_mask = block_mask.reshape(nb_r, nb_c)
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
                _, prune_col_idx = torch.topk(col_imp, n_prune_cols, largest=False)
                col_mask = torch.zeros(nb_c, dtype=torch.bool, device=W.device)
                col_mask.scatter_(0, prune_col_idx, True)
                block_mask = col_mask.unsqueeze(0).expand(nb_r, -1).contiguous()
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