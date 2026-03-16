import os

import numpy as np
import torch
from transformers import (
    LongformerModel,
    LongformerTokenizer,
)

model_name = "allenai/longformer-base-4096"

tokenizer = LongformerTokenizer.from_pretrained(model_name)
model = LongformerModel.from_pretrained(model_name, output_hidden_states=True)

model.eval()

text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

L = 0
X = outputs.hidden_states[L][0]

layer = model.encoder.layer[L].attention.self

W_Q = layer.query.weight
W_K = layer.key.weight
W_V = layer.value.weight

W = torch.cat([W_Q, W_K, W_V], dim=1)

out_dir = "configs/sparta/inputs/longformer"
os.makedirs(out_dir, exist_ok=True)

np.save(f"{out_dir}/X.npy", X.detach().cpu().numpy())
np.save(f"{out_dir}/W.npy", W.detach().cpu().numpy())
