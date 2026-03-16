import os

import numpy as np
import torch
from transformers import (
    BertModel,
    BertTokenizer,
)

model_name = "bert-base-uncased"

print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(model_name)

print("Loading model...")
model = BertModel.from_pretrained(model_name, output_hidden_states=True)
model.eval()

text = "The quick brown fox jumps over the lazy dog"

print("Tokenizing...")
inputs = tokenizer(text, return_tensors="pt")

print("Running forward pass...")
with torch.no_grad():
    outputs = model(**inputs)

print("Extracting weights...")

L = 0  # layer index

# hidden states before attention
X = outputs.hidden_states[L][0]  # [seq_len, d_model]

layer = model.encoder.layer[L].attention.self

W_Q = layer.query.weight
W_K = layer.key.weight
W_V = layer.value.weight

# concatenate QKV like SparTA expects
W = torch.cat([W_Q, W_K, W_V], dim=1)  # [d_model, 3*d_model]

out_dir = "configs/sparta/inputs/bert-base"
os.makedirs(out_dir, exist_ok=True)

np.save(f"{out_dir}/X.npy", X.detach().cpu().numpy())
np.save(f"{out_dir}/W.npy", W.detach().cpu().numpy())

print("Saved matrices to:", out_dir)
