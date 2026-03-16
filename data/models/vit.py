import os

import numpy as np
import torch
from PIL import Image
from transformers import (
    ViTImageProcessor,
    ViTModel,
)

model_name = "google/vit-base-patch16-224"

processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTModel.from_pretrained(model_name, output_hidden_states=True)

model.eval()

image = Image.new("RGB", (224, 224), color="white")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

L = 0
X = outputs.hidden_states[L][0]

layer = model.encoder.layer[L].attention.attention

W_Q = layer.query.weight
W_K = layer.key.weight
W_V = layer.value.weight

W = torch.cat([W_Q, W_K, W_V], dim=1)

out_dir = "configs/sparta/inputs/vit"
os.makedirs(out_dir, exist_ok=True)

np.save(f"{out_dir}/X.npy", X.detach().cpu().numpy())
np.save(f"{out_dir}/W.npy", W.detach().cpu().numpy())
