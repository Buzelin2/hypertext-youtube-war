#!/usr/bin/env python3
"""Recreate the channel embedding projection used in notebooks/images/embedding_map.pdf."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import AutoMinorLocator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    import umap.umap_ as umap
except Exception:
    umap = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIRS = {
    "Iran": PROJECT_DIR / "outputs" / "iran" / "polar_youtube_run",
    "Afghanistan": PROJECT_DIR / "outputs" / "afghanistan" / "polar_youtube_run",
}
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "notebooks" / "images" / "embedding_map.pdf"


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where((norms == 0) | (~np.isfinite(norms)), 1.0, norms)
    return matrix / norms


def confidence_ellipse(
    ax,
    x: Iterable[float],
    y: Iterable[float],
    color: str,
    n_std: float = 1.55,
    alpha: float = 0.10,
    lw: float = 2.0,
) -> None:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    if len(x) < 3:
        return

    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-9))
    ax.add_patch(
        Ellipse(
            xy=(np.mean(x), np.mean(y)),
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            linewidth=lw,
            alpha=alpha,
            zorder=1,
        )
    )


def infer_channel_type(row: Dict[str, object]) -> str:
    source_text = " ".join(
        str(row.get(key, "")).lower()
        for key in ["source_kind", "label_majority", "group_label", "channel_title", "user_id"]
    )
    return "News" if "news" in source_text else "Influencer"


def validate_run_dir(conflict_name: str, run_dir: Path) -> None:
    required = [
        run_dir / "users.json",
        run_dir / "model" / "config.json",
        run_dir / "model" / "tokenizer.json",
        run_dir / "model" / "model.safetensors",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"{conflict_name} run directory is missing files needed for the embedding plot:\n"
            f"{missing_lines}\n\n"
            "Copy the trained Hugging Face model artifacts into the run directory, "
            "or pass --iran-run-dir/--afghanistan-run-dir pointing at the original run outputs."
        )


def load_channel_embeddings(conflict_name: str, run_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    validate_run_dir(conflict_name, run_dir)
    users = json.loads((run_dir / "users.json").read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(run_dir / "model")
    model = AutoModelForMaskedLM.from_pretrained(run_dir / "model")
    model.eval()

    with torch.no_grad():
        weights = model.get_input_embeddings().weight.detach().cpu().numpy().astype("float32")
    weights = l2_normalize_rows(weights)
    vocab = tokenizer.get_vocab()

    rows: List[Dict[str, object]] = []
    vectors: List[np.ndarray] = []
    for row in users:
        token = row.get("token")
        token_id = vocab.get(token)
        if token_id is None:
            continue

        rows.append(
            {
                "conflict": conflict_name,
                "channel_type": infer_channel_type(row),
                "channel_id": row.get("channel_id"),
                "channel_title": row.get("channel_title"),
                "user_id": row.get("user_id"),
                "token": token,
                "n_posts": int(row.get("n_posts", 0) or 0),
                "n_videos": int(row.get("n_videos", 0) or 0),
            }
        )
        vectors.append(weights[token_id])

    if not vectors:
        raise ValueError(f"No channel user-token vectors were found for {conflict_name}: {run_dir}")
    return pd.DataFrame(rows), np.vstack(vectors)


def build_embedding_frame(run_dirs: Dict[str, Path], seed: int) -> Tuple[pd.DataFrame, str]:
    frames = []
    vector_blocks = []

    for conflict_name, run_dir in run_dirs.items():
        meta_block, vector_block = load_channel_embeddings(conflict_name, run_dir)
        frames.append(meta_block)
        vector_blocks.append(vector_block)

    embedding_df = pd.concat(frames, ignore_index=True)
    x_matrix = l2_normalize_rows(np.vstack(vector_blocks))

    pca_dims = max(2, min(25, x_matrix.shape[0] - 1, x_matrix.shape[1]))
    x_pca = PCA(n_components=pca_dims, random_state=seed).fit_transform(x_matrix)

    if umap is not None:
        n_neighbors = min(10, x_matrix.shape[0] - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.45,
            metric="cosine",
            init=x_pca[:, :2],
            random_state=seed,
        )
        coords = reducer.fit_transform(x_matrix)
        reducer_label = f"UMAP (cosine metric, PCA init, n_neighbors={n_neighbors})"
    else:
        perplexity = max(5, min(12, (x_matrix.shape[0] - 1) // 3))
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            metric="cosine",
            random_state=seed,
        ).fit_transform(x_pca)
        reducer_label = f"t-SNE fallback (cosine metric, perplexity={perplexity})"

    embedding_df["x"] = coords[:, 0]
    embedding_df["y"] = coords[:, 1]
    embedding_df["size"] = 110 + 24 * np.log1p(embedding_df["n_posts"].clip(lower=1))
    return embedding_df, reducer_label


def build_shift_frame(embedding_df: pd.DataFrame) -> pd.DataFrame:
    shared_rows = []
    for channel_id, group in embedding_df.groupby("channel_id"):
        if group["conflict"].nunique() != 2:
            continue
        group = group.sort_values("conflict").reset_index(drop=True)
        shared_rows.append(
            {
                "channel_id": channel_id,
                "channel_title": group.loc[0, "channel_title"],
                "proj_shift": float(
                    np.hypot(group.loc[0, "x"] - group.loc[1, "x"], group.loc[0, "y"] - group.loc[1, "y"])
                ),
            }
        )
    return pd.DataFrame(shared_rows, columns=["channel_id", "channel_title", "proj_shift"]).sort_values(
        "proj_shift", ascending=False
    )


def plot_embedding_map(
    embedding_df: pd.DataFrame,
    reducer_label: str,
    output_path: Path,
    label_all: bool = False,
    figure_width: float = 20,
    figure_height: float = 8,
    dpi: int = 220,
) -> None:
    conflict_colors = {"Iran": "#c44e52", "Afghanistan": "#4c72b0"}
    marker_map = {"News": "o", "Influencer": "^"}
    shift_df = build_shift_frame(embedding_df)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfcfe")

    x_span = embedding_df["x"].max() - embedding_df["x"].min()
    y_span = embedding_df["y"].max() - embedding_df["y"].min()
    x_pad = 0.10 * x_span if x_span > 0 else 1.0
    y_pad = 0.10 * y_span if y_span > 0 else 1.0
    ax.set_xlim(embedding_df["x"].min() - x_pad, embedding_df["x"].max() + x_pad)
    ax.set_ylim(embedding_df["y"].min() - y_pad, embedding_df["y"].max() + y_pad)

    for _, row in shift_df.iterrows():
        pair = embedding_df[embedding_df["channel_id"].eq(row["channel_id"])].sort_values("conflict")
        ax.plot(
            pair["x"],
            pair["y"],
            color="#7f7f7f",
            lw=1.2,
            alpha=0.58,
            zorder=1,
            solid_capstyle="round",
        )

    for conflict, color in conflict_colors.items():
        sub = embedding_df[embedding_df["conflict"].eq(conflict)]
        confidence_ellipse(ax, sub["x"], sub["y"], color=color)
        ax.scatter(
            sub["x"].mean(),
            sub["y"].mean(),
            s=340,
            marker="X",
            c=color,
            edgecolors="white",
            linewidths=1.9,
            zorder=4,
        )

    for conflict in ["Iran", "Afghanistan"]:
        for channel_type in ["News", "Influencer"]:
            sub = embedding_df[
                embedding_df["conflict"].eq(conflict) & embedding_df["channel_type"].eq(channel_type)
            ]
            ax.scatter(
                sub["x"],
                sub["y"],
                s=sub["size"],
                c=conflict_colors[conflict],
                marker=marker_map[channel_type],
                edgecolors="black",
                linewidths=0.9,
                alpha=1.0,
                zorder=3,
            )

    if label_all:
        text_dx = 0.012 * x_span if x_span > 0 else 0.05
        text_dy = 0.012 * y_span if y_span > 0 else 0.05
        for _, row in embedding_df.iterrows():
            label = str(row["channel_title"]) if pd.notna(row["channel_title"]) else str(row["user_id"])
            ax.text(
                row["x"] + text_dx,
                row["y"] + text_dy,
                label,
                fontsize=9,
                color="black",
                alpha=0.95,
                zorder=5,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.65),
            )

    ax.text(
        0.015,
        0.02,
        reducer_label,
        transform=ax.transAxes,
        fontsize=16,
        color="#444444",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="#d0d7de",
            linewidth=0.9,
            alpha=0.97,
        ),
    )

    ax.set_xlabel("Projection axis 1", fontsize=22, fontweight="bold")
    ax.set_ylabel("Projection axis 2", fontsize=22, fontweight="bold")
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(which="major", color="#b8c0cc", linewidth=0.85, alpha=0.48)
    ax.grid(which="minor", color="#d7dde6", linewidth=0.60, alpha=0.60)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", labelsize=18, width=1.2, length=7.5)
    ax.tick_params(axis="both", which="minor", width=0.8, length=4.5)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#b8bec9")
        ax.spines[spine].set_linewidth(1.0)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=conflict_colors["Iran"],
               markeredgecolor="black", markeredgewidth=0.8, markersize=11, label="Iran"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=conflict_colors["Afghanistan"],
               markeredgecolor="black", markeredgewidth=0.8, markersize=11, label="Afghanistan"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=1.0, markersize=11, label="News"),
        Line2D([0], [0], marker="^", linestyle="", markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=1.0, markersize=11.5, label="Influencer"),
        Line2D([0], [0], color="#7f7f7f", lw=1.3, alpha=0.75, label="Same channel across wars"),
        Line2D([0], [0], marker="X", linestyle="", markerfacecolor="#666666",
               markeredgecolor="white", markeredgewidth=1.0, markersize=12, label="Conflict centroid"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        borderaxespad=0.0,
        frameon=True,
        fancybox=True,
        framealpha=0.97,
        facecolor="white",
        edgecolor="#d0d7de",
        fontsize=18,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iran-run-dir", type=Path, default=DEFAULT_RUN_DIRS["Iran"])
    parser.add_argument("--afghanistan-run-dir", type=Path, default=DEFAULT_RUN_DIRS["Afghanistan"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-all", action="store_true", help="Render every channel label.")
    parser.add_argument("--figure-width", type=float, default=20)
    parser.add_argument("--figure-height", type=float, default=8)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = {
        "Iran": args.iran_run_dir.expanduser().resolve(),
        "Afghanistan": args.afghanistan_run_dir.expanduser().resolve(),
    }
    embedding_df, reducer_label = build_embedding_frame(run_dirs, seed=args.seed)
    plot_embedding_map(
        embedding_df=embedding_df,
        reducer_label=reducer_label,
        output_path=args.output.expanduser().resolve(),
        label_all=args.label_all,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        dpi=args.dpi,
    )
    print(f"Saved figure to: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
