# testing multiple CLIP model variants to see which is most useful for lost and found matching

import os
import re
import time
import torch
import open_clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
IMAGE_DIR = "images"
HEATMAP_DIR = "new_heatmaps"

NEW_IMAGE_DIR = "new_testcases"
RESULTS_DIR = "csvRes"

# (model name, pretrained tag, display label)
MODELS_TO_TEST = [
    ("ViT-B-32", "openai", "B-32 OpenAI"),
    ("ViT-L-14", "openai", "L-14 OpenAI"),
    ("ViT-B-32", "laion2b_s34b_b79k", "B-32 LAION2B"),
    ("ViT-B-16", "openai", "B-16 OpenAI"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")


def get_item_id(filename):
    # gets item from image filename
    # wallet_1 and wallet_2 are separate item from wallet2_1 and wallet2_2
    match = re.match(r"([a-zA-Z]+\d*)_", filename)
    return match.group(1) if match else filename


def load_image_list(image_dir):
    return [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]


def embed_all_images(model, preprocess, image_dir, filenames):
    # this also measures the timing info
    embeddings = {}
    per_image_times = []

    for f in filenames:
        path = os.path.join(image_dir, f)
        img = Image.open(path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            features = model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        per_image_times.append(elapsed)
        embeddings[f] = features.cpu().numpy().flatten()

    timing = {
        "total_embed_time": sum(per_image_times),
        "avg_time_per_image": sum(per_image_times) / len(per_image_times),
    }
    return embeddings, timing


def rank_all(query_name, embeddings):
    query_emb = embeddings[query_name].reshape(1, -1)
    results = []
    for name, emb in embeddings.items():
        if name == query_name:
            continue
        sim = cosine_similarity(query_emb, emb.reshape(1, -1))[0][0]
        results.append((name, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def classifyNew(new_embed, old_embed):
    res = []

    for nn, ne in new_embed.items():
        bm = None
        bs = -1

        nv = ne.reshape(1, -1)

        for on, oe in old_embed.items():
            simi = cosine_similarity(nv, oe.reshape(1, -1))[0][0]
            if simi > bs:
                bs = simi
                bm = on

        predic = get_item_id(bm)

        res.append({
            "test_image": nn,
            "predicted_item": predic,
            "matched_image": bm,
            "similarity": bs
        })

    return res

def csvOut(res, path):
    with open(path, "w", newline = "") as file:
        w = csv.DictWriter(file,
            fieldnames=[
                "test_image",
                "predicted_item",
                "matched_image",
                "similarity"
            ]
        )
        w.writeheader()

        for r in res:
            w.writerow({
                "test_image": r["test_image"],
                "predicted_item": r["predicted_item"],
                "matched_image": r["matched_image"],
                "similarity": f"{r['similarity']:.4f}"
            })

    print(f"saved classification results to {path}")

def evaluate_model(embeddings, filenames):
    # for every image, check whether the top-1 and top-3 ranked matches are same item
    top1_hits = 0
    top3_hits = 0
    total = 0

    for query_name in filenames:
        query_id = get_item_id(query_name)
        ranked = rank_all(query_name, embeddings)

        top1_id = get_item_id(ranked[0][0])
        top3_ids = [get_item_id(n) for n, _ in ranked[:3]]

        if top1_id == query_id:
            top1_hits += 1
        if query_id in top3_ids:
            top3_hits += 1
        total += 1

    return {
        "top1_accuracy": top1_hits / total,
        "top3_accuracy": top3_hits / total,
        "total_queries": total,
    }


def build_similarity_matrix(embeddings, filenames):
    # cosine similarity matrix across every image
    n = len(filenames)
    matrix = np.zeros((n, n))
    for i, n1 in enumerate(filenames):
        for j, n2 in enumerate(filenames):
            matrix[i][j] = cosine_similarity(
                embeddings[n1].reshape(1, -1), embeddings[n2].reshape(1, -1)
            )[0][0]
    return matrix


def plot_heatmap(matrix, filenames, title, save_path):
    order = sorted(range(len(filenames)), key=lambda i: get_item_id(filenames[i]))
    sorted_names = [filenames[i] for i in order]
    sorted_matrix = matrix[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sorted_matrix,
        xticklabels=sorted_names,
        yticklabels=sorted_names,
        cmap="viridis",
        vmin=0, vmax=1,
        square=True,
        cbar_kws={"label": "cosine similarity"},
        ax=ax,
    )
    ax.set_title(title)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved heatmap to {save_path}")


if __name__ == "__main__":
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    filenames = load_image_list(IMAGE_DIR)
    nf = load_image_list(NEW_IMAGE_DIR)
    print(f"found {len(filenames)} images in '{IMAGE_DIR}'")
    print(f"found {len(nf)} images in '{NEW_IMAGE_DIR}'")

    item_ids = sorted(set(get_item_id(f) for f in filenames))
    print(f"detected {len(item_ids)} distinct items: {item_ids}\n")

    model_results = []

    for model_name, pretrained, label in MODELS_TO_TEST:
        print(f"=== {label} ===")
        print(f"loading {model_name} ({pretrained})...")
        load_start = time.perf_counter()
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device).eval()
        load_time = time.perf_counter() - load_start

        embeddings, timing = embed_all_images(model, preprocess, IMAGE_DIR, filenames)
        print(f"embedded {len(embeddings)} images")
        print(f"model load time: {load_time:.2f}s")
        print(f"avg embed time per image: {timing['avg_time_per_image']*1000:.1f}ms")
        print(f"total embed time: {timing['total_embed_time']:.2f}s")

        ne, timing = embed_all_images(model, preprocess, NEW_IMAGE_DIR, nf)
        print(f"embedded {len(ne)} images")
        print(f"model load time: {load_time:.2f}s")
        print(f"avg embed time per image: {timing['avg_time_per_image']*1000:.1f}ms")
        print(f"total embed time: {timing['total_embed_time']:.2f}s")

        c_res = classifyNew(ne, embeddings)
        sl= label.replace(" ", "_").replace("-", "_")
        c_path = os.path.join(RESULTS_DIR, f"{sl}_classily.csv")
        csvOut(c_res, c_path)

        scores = evaluate_model(embeddings, filenames)
        print(f"top-1 accuracy: {scores['top1_accuracy']:.2%}")
        print(f"top-3 accuracy: {scores['top3_accuracy']:.2%}")

        # make heatmap
        matrix = build_similarity_matrix(embeddings, filenames)
        safe_label = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_")
        heatmap_path = os.path.join(HEATMAP_DIR, f"{safe_label}.png")
        plot_heatmap(matrix, filenames, f"{label} — pairwise cosine similarity", heatmap_path)
        print()

        model_results.append({
            "label": label,
            "model_name": model_name,
            "pretrained": pretrained,
            "load_time": load_time,
            "matrix": matrix,
            **timing,
            **scores,
        })

        # free up memory before loading the next model
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # final leaderboard, best model first
    model_results.sort(key=lambda r: r["top1_accuracy"], reverse=True)

    print("=" * 60)
    print("MODEL LEADERBOARD (ranked by top-1 accuracy)")
    print("=" * 60)
    for rank, r in enumerate(model_results, start=1):
        print(f"{rank}. {r['label']}")
        print(f"   top-1: {r['top1_accuracy']:.2%}   top-3: {r['top3_accuracy']:.2%}")
        print(f"   avg embed time/image: {r['avg_time_per_image']*1000:.1f}ms   model load: {r['load_time']:.2f}s")

    print("\n" + "=" * 60)
    print("EFFICIENCY LEADERBOARD (accuracy per ms of embed time)")
    print("=" * 60)
    efficiency_ranked = sorted(
        model_results,
        key=lambda r: r["top1_accuracy"] / (r["avg_time_per_image"] * 1000),
        reverse=True,
    )
    for rank, r in enumerate(efficiency_ranked, start=1):
        efficiency = r["top1_accuracy"] / (r["avg_time_per_image"] * 1000)
        print(f"{rank}. {r['label']}  ({efficiency:.4f} accuracy/ms)")

    # heatmap comparison image with all the models we usedd
    fig, axes = plt.subplots(1, len(model_results), figsize=(7 * len(model_results), 6))
    if len(model_results) == 1:
        axes = [axes]
    order = sorted(range(len(filenames)), key=lambda i: get_item_id(filenames[i]))
    sorted_names = [filenames[i] for i in order]
    for ax, r in zip(axes, model_results):
        sorted_matrix = r["matrix"][np.ix_(order, order)]
        sns.heatmap(sorted_matrix, xticklabels=sorted_names, yticklabels=sorted_names,
                    cmap="viridis", vmin=0, vmax=1, square=True, cbar=False, ax=ax)
        ax.set_title(f"{r['label']}\ntop-1: {r['top1_accuracy']:.0%}", fontsize=10)
        ax.tick_params(labelsize=6)
        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.setp(ax.get_yticklabels(), rotation=0)

        #debug: get raw values
        # print(f"\npairwise similarity matrix ({r['label']}):")
        # print("        " + "  ".join(f"{n[:8]:>8}" for n in sorted_names))
        # for i, n in enumerate(sorted_names):
        #     print(f"{n[:10]:>10}" + "  ".join(f"{sorted_matrix[i][j]:.3f}" for j in range(len(sorted_names))))

    plt.tight_layout()
    combined_path = os.path.join(HEATMAP_DIR, "all_models_comparison.png")
    plt.savefig(combined_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved side-by-side comparison to {combined_path}")
