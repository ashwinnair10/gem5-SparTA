import os

import numpy as np
import torch
from transformers import (
    GPT2Model,
    GPT2Tokenizer,
)

model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)

model.eval()

text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

L = 0
X = outputs.hidden_states[L][0]

layer = model.h[L].attn

W = layer.c_attn.weight  # already [d_model, 3*d_model]

out_dir = "configs/sparta/inputs/gpt2"
os.makedirs(out_dir, exist_ok=True)

np.save(f"{out_dir}/X.npy", X.detach().cpu().numpy())
np.save(f"{out_dir}/W.npy", W.detach().cpu().numpy())
