import os

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
)

# =============================
# CHANGE THIS MODEL NAME
# =============================
model_name = "Intel/bert-base-uncased-sparse-90-unstructured-pruneofa"
layer_index = 0

# =============================
# Create output directory
# =============================
out_dir = f"configs/sparta/inputs/bert-90-sparse"
os.makedirs(out_dir, exist_ok=True)

# =============================
# Load model + tokenizer
# =============================
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

model.eval()

# =============================
# Generate real activation X
# =============================
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Hidden states BEFORE attention
X = outputs.hidden_states[layer_index][0]  # [seq_len, d_model]

# =============================
# Extract Q, K, V weights
# =============================
layer = model.encoder.layer[layer_index].attention.self

W_Q = layer.query.weight
W_K = layer.key.weight
W_V = layer.value.weight

# Concatenate for simulator
W = torch.cat([W_Q, W_K, W_V], dim=1)  # [d_model, 3*d_model]

# =============================
# Save matrices
# =============================
X_np = X.detach().cpu().numpy()
W_np = W.detach().cpu().numpy()

np.save(os.path.join(out_dir, "X.npy"), X_np)
np.save(os.path.join(out_dir, "W.npy"), W_np)

# =============================
# Print info
# =============================
print("Saved to:", out_dir)
print("X shape:", X_np.shape)
print("W shape:", W_np.shape)

sparsity = 1.0 - (np.count_nonzero(W_np) / W_np.size)
print("W sparsity:", sparsity)
