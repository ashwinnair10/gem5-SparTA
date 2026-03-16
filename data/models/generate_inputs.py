import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
)


def prune_weights(W, sparsity):
    if sparsity == 0:
        return W

    threshold = np.percentile(np.abs(W), sparsity * 100)
    W[np.abs(W) < threshold] = 0
    return W


def get_workload_text(workload):

    if workload == "glue":
        ds = load_dataset("glue", "sst2")
        return ds["train"][0]["sentence"]

    if workload == "squad":
        ds = load_dataset("squad")
        return ds["train"][0]["context"]

    if workload == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")

        for row in ds["train"]:
            txt = row["text"].strip()
            if len(txt) > 20:
                return txt

    if workload == "imdb":
        ds = load_dataset("imdb")
        return ds["train"][0]["text"]

    raise ValueError("Unknown workload")


def get_image_sample():

    ds = load_dataset("cifar10")
    return ds["train"][0]["img"]


def extract_qkv(model, model_name, layer_id=0):

    # ---------- BERT / RoBERTa / BigBird / Longformer ----------
    if hasattr(model, "encoder") and hasattr(
        model.encoder.layer[0].attention, "self"
    ):

        layer = model.encoder.layer[layer_id].attention.self

        W_Q = layer.query.weight
        W_K = layer.key.weight
        W_V = layer.value.weight

        W = torch.cat([W_Q, W_K, W_V], dim=1)
        return W.detach().cpu().numpy()

    # ---------- ViT ----------
    if hasattr(model, "encoder") and hasattr(
        model.encoder.layer[0].attention, "attention"
    ):

        layer = model.encoder.layer[layer_id].attention.attention

        W_Q = layer.query.weight
        W_K = layer.key.weight
        W_V = layer.value.weight

        W = torch.cat([W_Q, W_K, W_V], dim=1)
        return W.detach().cpu().numpy()

    # ---------- GPT ----------
    if hasattr(model, "h"):

        layer = model.h[layer_id].attn
        W = layer.c_attn.weight.T
        return W.detach().cpu().numpy()

    raise RuntimeError("Unsupported architecture")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)

    parser.add_argument(
        "--workload",
        default="glue",
        choices=["glue", "squad", "wikitext", "imdb", "image"],
    )

    parser.add_argument("--sparsity", type=float, default=0)

    args = parser.parse_args()

    print("Loading model...")

    model = AutoModel.from_pretrained(args.model, output_hidden_states=True)

    model.eval()

    # ---------- TEXT MODELS ----------
    if args.workload != "image":

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        print("Loading workload text...")
        text = get_workload_text(args.workload)

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )

    # ---------- VISION MODELS ----------
    else:

        processor = AutoImageProcessor.from_pretrained(args.model)

        print("Loading image sample...")
        image = get_image_sample()

        inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    L = 0
    X = outputs.hidden_states[L][0]

    print("Extracting QKV weights...")
    W = extract_qkv(model, args.model, L)

    print("Applying pruning...")
    W = prune_weights(W, args.sparsity)

    sparsity_tag = (
        "dense" if args.sparsity == 0 else f"{int(args.sparsity*100)}"
    )

    safe_model_name = args.model.replace("/", "_")

    out_dir = f"configs/sparta/inputs/{safe_model_name}_{args.workload}_{sparsity_tag}"

    os.makedirs(out_dir, exist_ok=True)

    np.save(f"{out_dir}/X.npy", X.detach().cpu().numpy())
    np.save(f"{out_dir}/W.npy", W)

    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
