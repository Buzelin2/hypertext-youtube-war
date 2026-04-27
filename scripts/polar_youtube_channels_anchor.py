#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from channel_embedding_common import (
    build_user_vectors,
    l2n,
    load_tokenizer_and_embeddings,
    load_users,
    resolve_config_path,
    resolve_model_and_user_paths,
    word_vec,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "polar_youtube_run"
DEFAULT_ANCHORS_PATH = PROJECT_DIR / "resources" / "anchors" / "anchors_war_youtube.example.json"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "anchor_config_youtube_channels.example.json"


@dataclass
class Cfg:
    out_dir: str = str(DEFAULT_OUTPUT_DIR)
    model_dir: Optional[str] = None
    users_csv: Optional[str] = None
    users_json: Optional[str] = None
    meta_json: Optional[str] = None
    anchors_path: str = str(DEFAULT_ANCHORS_PATH)
    usr_prefix_fallback: str = "usr"
    min_posts_default: int = 1
    seed: int = 123
    print_every: int = 500
    save_csv: bool = True

    @classmethod
    def from_json(cls, path: str) -> "Cfg":
        config_path = Path(path).expanduser().resolve()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"Config JSON must be an object, got {type(data).__name__}.")

        def _resolve_str_path(value: Optional[str]) -> Optional[str]:
            return resolve_config_path(str(config_path), value)

        for key in ("out_dir", "model_dir", "users_csv", "users_json", "meta_json", "anchors_path"):
            if key in data:
                data[key] = _resolve_str_path(data[key])
        return cls(**data)

def _parse_anchor_terms(raw: Any) -> Tuple[List[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    terms: List[str] = []

    if isinstance(raw, str):
        terms = [raw]
    elif isinstance(raw, list):
        terms = [x for x in raw if isinstance(x, str)]
    elif isinstance(raw, dict):
        terms_value = raw.get("terms")
        if isinstance(terms_value, str):
            terms = [terms_value]
        elif isinstance(terms_value, list):
            terms = [x for x in terms_value if isinstance(x, str)]
        for key in ("label", "description", "group"):
            if key in raw:
                meta[key] = raw[key]
    else:
        raise TypeError(f"Anchor spec must be a string, list, or object, got {type(raw).__name__}.")

    cleaned = [t.strip() for t in terms if isinstance(t, str) and t.strip()]
    return cleaned, meta


def run(cfg: Cfg) -> None:
    np.random.seed(cfg.seed)

    model_dir, users_csv, users_json, meta_json = resolve_model_and_user_paths(
        cfg.out_dir, cfg.model_dir, cfg.users_csv, cfg.users_json, cfg.meta_json
    )
    users = load_users(users_csv, users_json, cfg.min_posts_default, meta_json, cfg.usr_prefix_fallback)
    if not users:
        raise SystemExit("No users after filtering.")

    anchors_raw = json.loads(Path(cfg.anchors_path).read_text(encoding="utf-8"))
    if not isinstance(anchors_raw, dict):
        raise TypeError("Anchors JSON must be an object keyed by anchor name.")

    print(f"users={len(users)}")

    tok, W, vocab = load_tokenizer_and_embeddings(model_dir)

    cache: Dict[str, Optional[np.ndarray]] = {}
    anchor_vecs: Dict[str, Tuple[np.ndarray, int, Dict[str, Any]]] = {}
    anchor_diagnostics: List[Dict[str, Any]] = []
    for anchor_name, spec in anchors_raw.items():
        terms, extra_meta = _parse_anchor_terms(spec)
        term_vecs = [v for term in terms if (v := word_vec(term, tok, W, cache)) is not None]
        if term_vecs:
            centroid = l2n(np.vstack(term_vecs).mean(axis=0).astype("float32"))
            anchor_vecs[anchor_name] = (centroid, len(term_vecs), extra_meta)
        anchor_diagnostics.append(
            {
                "anchor": anchor_name,
                "label": extra_meta.get("label"),
                "description": extra_meta.get("description"),
                "group": extra_meta.get("group"),
                "requested_terms": terms,
                "resolved_terms": len(term_vecs),
            }
        )
        print(f"[{anchor_name}] requested={len(terms)} resolved={len(term_vecs)}")

    user_vecs, missing = build_user_vectors(users, vocab, W)

    if not user_vecs:
        raise SystemExit("No user tokens found in the saved model vocabulary.")
    if missing:
        print(f"Missing {len(missing)} user tokens (showing up to 5): {missing[:5]}")

    rows: List[Dict[str, Any]] = []
    uids = list(user_vecs.keys())
    for aidx, (anchor_name, payload) in enumerate(anchor_vecs.items(), 1):
        anchor_vec, n_terms, extra_meta = payload
        for k, uid in enumerate(uids, 1):
            meta = users[uid]
            similarity = float(np.dot(user_vecs[uid], anchor_vec))
            rows.append(
                {
                    "user_id": uid,
                    "anchor": anchor_name,
                    "anchor_label": extra_meta.get("label") or anchor_name,
                    "anchor_group": extra_meta.get("group"),
                    "anchor_description": extra_meta.get("description"),
                    "cosine_similarity": similarity,
                    "cosine_distance": float(1.0 - similarity),
                    "n_anchor_terms": n_terms,
                    "n_posts": int(meta.get("n_posts", 0)),
                    "label_majority": meta.get("label_majority"),
                    "targets": meta.get("targets"),
                    "channel_id": meta.get("channel_id"),
                    "channel_title": meta.get("channel_title"),
                    "group_label": meta.get("group_label"),
                    "corpus": meta.get("corpus"),
                    "source_kind": meta.get("source_kind"),
                    "n_videos": meta.get("n_videos"),
                }
            )
            if k % cfg.print_every == 0:
                print(f"anchor {aidx}/{len(anchor_vecs)} {k}/{len(uids)} users")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["similarity_zscore_within_anchor"] = (
            df.groupby("anchor", sort=False)["cosine_similarity"]
            .transform(lambda s: (s - s.mean()) / s.std(ddof=0) if float(s.std(ddof=0)) > 0 else 0.0)
        )
        df["distance_zscore_within_anchor"] = (
            df.groupby("anchor", sort=False)["cosine_distance"]
            .transform(lambda s: (s - s.mean()) / s.std(ddof=0) if float(s.std(ddof=0)) > 0 else 0.0)
        )
    else:
        df["similarity_zscore_within_anchor"] = []
        df["distance_zscore_within_anchor"] = []

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows = df.to_dict(orient="records")
    (out_dir / "per_user_anchor_scores.json").write_text(json.dumps(long_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.save_csv:
        df.to_csv(out_dir / "per_user_anchor_scores.csv", index=False)

    wide: Dict[str, Dict[str, Any]] = {}
    for uid, sub in df.groupby("user_id", sort=False):
        meta = users.get(uid, {}).copy()
        block = {
            "user_id": uid,
            "token": meta.get("token"),
            "n_posts": int(meta.get("n_posts", 0)),
            "channel_id": meta.get("channel_id"),
            "channel_title": meta.get("channel_title"),
            "group_label": meta.get("group_label"),
            "corpus": meta.get("corpus"),
            "source_kind": meta.get("source_kind"),
            "n_videos": meta.get("n_videos"),
            "anchors": {},
        }
        for _, row in sub.iterrows():
            block["anchors"][row["anchor"]] = {
                "anchor_label": row["anchor_label"],
                "anchor_group": row["anchor_group"],
                "anchor_description": row["anchor_description"],
                "cosine_similarity": float(row["cosine_similarity"]),
                "cosine_distance": float(row["cosine_distance"]),
                "similarity_zscore_within_anchor": float(row["similarity_zscore_within_anchor"]),
                "distance_zscore_within_anchor": float(row["distance_zscore_within_anchor"]),
                "n_anchor_terms": int(row["n_anchor_terms"]),
            }
        wide[uid] = block
    (out_dir / "per_user_anchor_scores_wide.json").write_text(json.dumps(list(wide.values()), ensure_ascii=False, indent=2), encoding="utf-8")

    anchor_summaries = []
    for anchor_name, sub in df.groupby("anchor", sort=False):
        sims = pd.to_numeric(sub["cosine_similarity"], errors="coerce")
        dists = pd.to_numeric(sub["cosine_distance"], errors="coerce")
        anchor_summaries.append(
            {
                "anchor": anchor_name,
                "n_users": int(sub.shape[0]),
                "mean_cosine_similarity": None if sims.dropna().empty else float(sims.mean()),
                "std_cosine_similarity": None if sims.dropna().empty else float(sims.std(ddof=0)),
                "min_cosine_similarity": None if sims.dropna().empty else float(sims.min()),
                "max_cosine_similarity": None if sims.dropna().empty else float(sims.max()),
                "mean_cosine_distance": None if dists.dropna().empty else float(dists.mean()),
            }
        )
    (out_dir / "anchor_summaries.json").write_text(json.dumps(anchor_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "anchor_diagnostics.json").write_text(json.dumps(anchor_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    run_summary = {
        "out_dir": str(out_dir),
        "model_dir": str(model_dir),
        "anchors_path": cfg.anchors_path,
        "n_users_requested": len(users),
        "n_users_scored": len(user_vecs),
        "n_anchors_requested": len(anchors_raw),
        "n_anchors_scored": len(anchor_vecs),
        "n_missing_user_tokens": len(missing),
        "missing_user_tokens": missing,
        "seed": cfg.seed,
        "files": {
            "per_user_anchor_scores_json": str(out_dir / "per_user_anchor_scores.json"),
            "per_user_anchor_scores_wide_json": str(out_dir / "per_user_anchor_scores_wide.json"),
            "anchor_summaries_json": str(out_dir / "anchor_summaries.json"),
            "anchor_diagnostics_json": str(out_dir / "anchor_diagnostics.json"),
            "per_user_anchor_scores_csv": str(out_dir / "per_user_anchor_scores.csv") if cfg.save_csv else None,
        },
    }
    (out_dir / "anchor_run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score YouTube channel embeddings against single-word or multi-term anchors.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to a JSON config file. Defaults to {DEFAULT_CONFIG_PATH}",
    )
    args = parser.parse_args()
    run(Cfg.from_json(args.config))


if __name__ == "__main__":
    main()
