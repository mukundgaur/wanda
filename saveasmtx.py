from transformers import AutoModelForCausalLM
import torch
import scipy.sparse
import scipy.io
import os

model = AutoModelForCausalLM.from_pretrained(
    "out/llama2_7b/unstructured/0.3",
    torch_dtype=torch.float16,
    device_map="cpu"
)

save_dir = "out/llama2_7b/unstructured/0.3/mtx"
os.makedirs(save_dir, exist_ok=True)

target_layers = {
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.15.self_attn.q_proj.weight",
    "model.layers.15.mlp.gate_proj.weight",
    "model.layers.31.self_attn.q_proj.weight",
    "model.layers.31.mlp.gate_proj.weight",
}

for name, param in model.named_parameters():
    if name in target_layers:
        w = param.detach().cpu().float().numpy()
        sparse_w = scipy.sparse.csr_matrix(w)
        filename = name.replace(".", "_") + ".mtx"
        scipy.io.mmwrite(os.path.join(save_dir, filename), sparse_w)
        print(f"Saved {name} | shape {w.shape} | nnz {sparse_w.nnz}")