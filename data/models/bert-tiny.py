import os

import numpy as np
import torch
from transformers import (
    BertModel,
    BertTokenizer,
)

model_name = "prajjwal1/bert-tiny"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_hidden_states=True)

model.eval()
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
L = 0  # attention layer index

# Hidden states BEFORE attention at layer L
X = outputs.hidden_states[L][0]  # shape: [seq_len, d_model]
layer = model.encoder.layer[L].attention.self

W_Q = layer.query.weight  # [d_model, d_model]
W_K = layer.key.weight
W_V = layer.value.weight

# Concatenate like your simulator expects
W = torch.cat([W_Q, W_K, W_V], dim=1)  # [d_model, 3*d_model]

os.makedirs("configs/sparta/inputs/bert-tiny", exist_ok=True)
np.save("configs/sparta/inputs/bert-tiny/X.npy", X.detach().cpu().numpy())
np.save("configs/sparta/inputs/bert-tiny/W.npy", W.detach().cpu().numpy())
