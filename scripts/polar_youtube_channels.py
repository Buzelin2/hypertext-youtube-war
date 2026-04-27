#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from channel_embedding_common import (
    bh_fdr,
    build_user_vectors,
    l2n,
    load_tokenizer_and_embeddings,
    load_users,
    perm_p,
    resolve_config_path,
    resolve_model_and_user_paths,
    word_vec,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "polar_youtube_run"
DEFAULT_ATTRIBUTES_PATH = PROJECT_DIR / "resources" / "attributes" / "attributes_war_youtube.example.json"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "polar_config_youtube_channels.example.json"


@dataclass
class Cfg:
    out_dir: str = str(DEFAULT_OUTPUT_DIR)
    model_dir: Optional[str] = None
    users_csv: Optional[str] = None
    users_json: Optional[str] = None
    meta_json: Optional[str] = None
    attributes_path: str = str(DEFAULT_ATTRIBUTES_PATH)
    usr_prefix_fallback: str = "usr"
    min_posts_default: int = 1
    alpha_bh: float = 0.05
    mc_samples: int = 2000
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

        for key in ("out_dir", "model_dir", "users_csv", "users_json", "meta_json", "attributes_path"):
            if key in data:
                data[key] = _resolve_str_path(data[key])
        return cls(**data)

def run(cfg: Cfg) -> None:
    np.random.seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)

    model_dir, users_csv, users_json, meta_json = resolve_model_and_user_paths(
        cfg.out_dir, cfg.model_dir, cfg.users_csv, cfg.users_json, cfg.meta_json
    )
    users = load_users(users_csv, users_json, cfg.min_posts_default, meta_json, cfg.usr_prefix_fallback)
    if not users:
        raise SystemExit("No users after filtering.")

    attrs = json.loads(Path(cfg.attributes_path).read_text(encoding="utf-8"))
    print(f"users={len(users)}")

    tok, W, vocab = load_tokenizer_and_embeddings(model_dir)

    cache: Dict[str, Optional[np.ndarray]] = {}
    pair_mats: Dict[str, Tuple[np.ndarray, np.ndarray, int, int, float]] = {}
    pair_diagnostics: List[Dict[str, Any]] = []
    for name, pair in attrs.items():
        pos = [w for w in (pair.get("pos", []) or []) if isinstance(w, str) and w.strip()]
        neg = [w for w in (pair.get("neg", []) or []) if isinstance(w, str) and w.strip()]
        A = [v for w in pos if (v := word_vec(w, tok, W, cache)) is not None]
        B = [v for w in neg if (v := word_vec(w, tok, W, cache)) is not None]
        A = np.vstack(A) if A else np.empty((0, W.shape[1]), "float32")
        B = np.vstack(B) if B else np.empty((0, W.shape[1]), "float32")
        sep = float(np.dot(l2n(A.mean(0)), l2n(B.mean(0)))) if len(A) > 0 and len(B) > 0 else np.nan
        pair_mats[name] = (A, B, len(A), len(B), sep)
        pair_diagnostics.append(
            {
                "pair": name,
                "n_pos_attr": len(A),
                "n_neg_attr": len(B),
                "centroid_cosine": None if not np.isfinite(sep) else float(sep),
                "pos_terms": pos,
                "neg_terms": neg,
            }
        )
        if len(A) > 0 and len(B) > 0:
            print(f"[{name}] pos={len(A)} neg={len(B)} cos(centroids)={sep:+.3f}")
        else:
            print(f"[{name}] pos={len(A)} neg={len(B)}")

    user_vecs, missing = build_user_vectors(users, vocab, W)

    if not user_vecs:
        raise SystemExit("No user tokens found in the saved model vocabulary.")
    if missing:
        print(f"Missing {len(missing)} user tokens (showing up to 5): {missing[:5]}")

    rows: List[Dict[str, Any]] = []
    uids = list(user_vecs.keys())
    t0 = time.time()
    for pidx, (pair_name, (A, B, m, n, sep)) in enumerate(pair_mats.items(), 1):
        if m == 0 or n == 0:
            for uid in uids:
                meta = users[uid]
                rows.append(
                    {
                        "user_id": uid,
                        "pair": pair_name,
                        "s": None,
                        "p_perm": None,
                        "n_posts": int(meta.get("n_posts", 0)),
                        "n_pos_attr": m,
                        "n_neg_attr": n,
                        "pair_centroid_cosine": None if not np.isfinite(sep) else float(sep),
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
            continue

        AB = np.vstack([A, B])
        for k, uid in enumerate(uids, 1):
            meta = users[uid]
            u = user_vecs[uid]
            d_all = AB @ u
            sd = float(np.std(d_all))
            s_obs = (d_all[:m].mean() - d_all[m:].mean()) / sd if sd > 0 and np.isfinite(sd) else np.nan
            p_val = perm_p(d_all, m, rng, s_obs, sd, cfg.mc_samples)
            rows.append(
                {
                    "user_id": uid,
                    "pair": pair_name,
                    "s": None if not np.isfinite(s_obs) else float(s_obs),
                    "p_perm": None if not np.isfinite(p_val) else float(p_val),
                    "n_posts": int(meta.get("n_posts", 0)),
                    "n_pos_attr": m,
                    "n_neg_attr": n,
                    "pair_centroid_cosine": None if not np.isfinite(sep) else float(sep),
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
                print(f"pair {pidx}/{len(pair_mats)} {k}/{len(uids)} users elapsed={time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df["signif_bh_fdr_0.05"] = False
    for _, sub in df.groupby("pair", sort=False):
        pvals = pd.to_numeric(sub["p_perm"], errors="coerce").to_numpy()
        df.loc[sub.index, "signif_bh_fdr_0.05"] = bh_fdr(pvals, cfg.alpha_bh)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows = df.to_dict(orient="records")
    (out_dir / "per_user_scores.json").write_text(json.dumps(long_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.save_csv:
        df.to_csv(out_dir / "per_user_scores.csv", index=False)

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
            "pairs": {},
        }
        for _, row in sub.iterrows():
            block["pairs"][row["pair"]] = {
                "s": None if pd.isna(row["s"]) else float(row["s"]),
                "p_perm": None if pd.isna(row["p_perm"]) else float(row["p_perm"]),
                "signif_bh_fdr_0.05": bool(row["signif_bh_fdr_0.05"]),
                "n_pos_attr": int(row["n_pos_attr"]),
                "n_neg_attr": int(row["n_neg_attr"]),
                "pair_centroid_cosine": None if pd.isna(row["pair_centroid_cosine"]) else float(row["pair_centroid_cosine"]),
            }
        wide[uid] = block
    (out_dir / "per_user_scores_wide.json").write_text(json.dumps(list(wide.values()), ensure_ascii=False, indent=2), encoding="utf-8")

    pair_summaries = []
    for pair_name, sub in df.groupby("pair", sort=False):
        s = pd.to_numeric(sub["s"], errors="coerce")
        pair_summaries.append(
            {
                "pair": pair_name,
                "n_users": int(sub.shape[0]),
                "n_significant": int(pd.Series(sub["signif_bh_fdr_0.05"]).fillna(False).sum()),
                "mean_s": None if s.dropna().empty else float(s.mean()),
                "std_s": None if s.dropna().empty else float(s.std(ddof=0)),
                "min_s": None if s.dropna().empty else float(s.min()),
                "max_s": None if s.dropna().empty else float(s.max()),
            }
        )
    (out_dir / "pair_summaries.json").write_text(json.dumps(pair_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "pair_diagnostics.json").write_text(json.dumps(pair_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    run_summary = {
        "out_dir": str(out_dir),
        "model_dir": str(model_dir),
        "attributes_path": cfg.attributes_path,
        "n_users_requested": len(users),
        "n_users_scored": len(user_vecs),
        "n_pairs": len(pair_mats),
        "n_missing_user_tokens": len(missing),
        "missing_user_tokens": missing,
        "alpha_bh": cfg.alpha_bh,
        "mc_samples": cfg.mc_samples,
        "seed": cfg.seed,
        "files": {
            "per_user_scores_json": str(out_dir / "per_user_scores.json"),
            "per_user_scores_wide_json": str(out_dir / "per_user_scores_wide.json"),
            "pair_summaries_json": str(out_dir / "pair_summaries.json"),
            "pair_diagnostics_json": str(out_dir / "pair_diagnostics.json"),
            "per_user_scores_csv": str(out_dir / "per_user_scores.csv") if cfg.save_csv else None,
        },
    }
    (out_dir / "polar_run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run POLAR scoring for YouTube channels treated as users.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to a JSON config file. Defaults to {DEFAULT_CONFIG_PATH}",
    )
    args = parser.parse_args()
    run(Cfg.from_json(args.config))


if __name__ == "__main__":
    main()
