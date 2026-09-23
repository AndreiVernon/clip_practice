# testing openai's CLIP model to see if it can be useful for lost and found project

import os
import torch
import open_clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path

# config
IMAGE_DIR = "images"
QUERY_IMAGE = r"images\bottle_1.jpg"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

# load model + preprocessing transform
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model = model.to(device).eval()

def embed_image(path):
    img = Image.open(path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)  # normalize for cosine sim
    return features.cpu().numpy().flatten()

def load_candidates(image_dir):
    files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    embeddings = {}
    for f in files:
        path = os.path.join(image_dir, f)
        embeddings[f] = embed_image(path)
    return embeddings

def rank_by_similarity(query_path, candidate_embeddings):
    query_emb = embed_image(query_path).reshape(1, -1)
    results = []
    for name, emb in candidate_embeddings.items():
        sim = cosine_similarity(query_emb, emb.reshape(1, -1))[0][0]
        results.append((name, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

if __name__ == "__main__":
    print(f"loading candidate images from '{IMAGE_DIR}'...")
    candidates = load_candidates(IMAGE_DIR)
    print(f"embedded {len(candidates)} images")

    print(f"\nranking against query image: {QUERY_IMAGE}")
    ranked = rank_by_similarity(QUERY_IMAGE, candidates)

    print("\nresults (highest similarity first):")
    for name, score in ranked:
        print(f"  {score:.4f}  {name}")

    # similarity matrix across all candidates - for heatmap
    names = list(candidates.keys())
    matrix = np.zeros((len(names), len(names)))
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            matrix[i][j] = cosine_similarity(
                candidates[n1].reshape(1, -1), candidates[n2].reshape(1, -1)
            )[0][0]

    print("\npairwise similarity matrix:")
    print("        " + "  ".join(f"{n[:6]:>6}" for n in names))
    for i, n in enumerate(names):
        print(f"{n[:8]:>8}" + "  ".join(f"{matrix[i][j]:.3f}" for j in range(len(names))))