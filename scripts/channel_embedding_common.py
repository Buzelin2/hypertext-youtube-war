#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def resolve_config_path(config_path: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (Path(config_path).expanduser().resolve().parent / candidate).resolve()
    return str(candidate)


def l2n_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[(norms == 0) | (~np.isfinite(norms))] = 1.0
    return matrix / norms


def l2n(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 or not np.isfinite(norm) else vector / norm


def word_vec(word: str, tok, embeddings: np.ndarray, cache: Dict[str, Optional[np.ndarray]]) -> Optional[np.ndarray]:
    if word in cache:
        return cache[word]
    normalized_word = word.lower() if getattr(tok, "do_lower_case", False) else word
    token_ids = tok.encode(normalized_word, add_special_tokens=False)
    cache[word] = None if not token_ids else l2n(embeddings[token_ids].mean(axis=0).astype("float32"))
    return cache[word]


def resolve_model_and_user_paths(
    out_dir: str,
    model_dir: Optional[str],
    users_csv: Optional[str],
    users_json: Optional[str],
    meta_json: Optional[str],
) -> Tuple[Path, Path, Optional[Path], Optional[Path]]:
    resolved_out_dir = Path(out_dir)
    resolved_model_dir = Path(model_dir) if model_dir else resolved_out_dir / "model"
    resolved_users_csv = Path(users_csv) if users_csv else resolved_out_dir / "users.csv"
    resolved_users_json = Path(users_json) if users_json else resolved_out_dir / "users.json"
    resolved_meta_json = Path(meta_json) if meta_json else resolved_out_dir / "meta.json"
    return resolved_model_dir, resolved_users_csv, resolved_users_json, resolved_meta_json


def load_users(
    users_csv: Path,
    users_json: Optional[Path],
    min_posts: int,
    meta_json: Optional[Path],
    fallback_prefix: str,
) -> Dict[str, Dict[str, Any]]:
    prefix = fallback_prefix
    try:
        if meta_json and meta_json.exists():
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            if isinstance(meta.get("usr_prefix"), str) and meta["usr_prefix"].strip():
                prefix = meta["usr_prefix"].strip()
            if "min_posts_per_user" in meta:
                min_posts = max(int(meta["min_posts_per_user"]), 1)
    except Exception:
        pass

    users: Dict[str, Dict[str, Any]] = {}

    if users_json and users_json.exists():
        try:
            rows = json.loads(users_json.read_text(encoding="utf-8"))
            if isinstance(rows, list):
                for row in rows:
                    uid = str((row or {}).get("user_id") or "").strip()
                    if not uid:
                        continue
                    try:
                        n_posts = int((row or {}).get("n_posts", 0))
                    except Exception:
                        n_posts = 0
                    if n_posts < min_posts:
                        continue
                    token = str((row or {}).get("token") or f"{prefix}{uid}")
                    users[uid] = dict(row)
                    users[uid]["token"] = token
                    users[uid]["n_posts"] = n_posts
                if users:
                    return users
        except Exception:
            pass

    with users_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        has_token_col = "token" in (reader.fieldnames or [])
        for row in reader:
            uid = (row.get("user_id") or "").strip()
            if not uid:
                continue
            try:
                n_posts = int(row.get("n_posts", "0"))
            except Exception:
                n_posts = 0
            if n_posts < min_posts:
                continue
            token = (row.get("token") or "").strip() if has_token_col else ""
            if not token:
                token = f"{prefix}{uid}"
            users[uid] = {
                "user_id": uid,
                "n_posts": n_posts,
                "token": token,
                "label_majority": (row.get("label_majority") or "").strip() or None,
                "targets": (row.get("targets") or "").strip() or None,
                "channel_id": (row.get("channel_id") or "").strip() or None,
                "channel_title": (row.get("channel_title") or "").strip() or None,
                "group_label": (row.get("group_label") or "").strip() or None,
                "corpus": (row.get("corpus") or "").strip() or None,
                "source_kind": (row.get("source_kind") or "").strip() or None,
                "n_videos": int(row.get("n_videos", "0") or 0),
            }
    return users


def load_tokenizer_and_embeddings(model_dir: Path):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForMaskedLM.from_pretrained(str(model_dir))
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight.detach().cpu().numpy().astype("float32")
    embeddings = l2n_rows(embeddings)
    vocab = tok.get_vocab()
    return tok, embeddings, vocab


def build_user_vectors(
    users: Dict[str, Dict[str, Any]],
    vocab: Dict[str, int],
    embeddings: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], list]:
    user_vecs: Dict[str, np.ndarray] = {}
    missing = []
    for uid, meta in users.items():
        token_id = vocab.get(meta["token"])
        if token_id is None:
            missing.append({"user_id": uid, "token": meta["token"]})
            continue
        user_vecs[uid] = l2n(embeddings[token_id])
    return user_vecs, missing


def bh_fdr(p_values: np.ndarray, alpha: float) -> np.ndarray:
    p = np.array(p_values, dtype=float, copy=True)
    p[np.isnan(p)] = np.inf
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    ranks = np.empty(n, int)
    ranks[order] = np.arange(1, n + 1)
    thresh = (ranks / n) * alpha
    passed = p <= thresh
    mx = (ranks * passed).max() if passed.any() else 0
    return (p <= (mx / n * alpha)) if mx > 0 else np.zeros(n, bool)


def perm_p(d_all: np.ndarray, m: int, rng: np.random.RandomState, s_obs: float, sd_all: float, mc: int) -> float:
    total_count = d_all.shape[0]
    n = total_count - m
    if m <= 0 or n <= 0 or not np.isfinite(s_obs) or sd_all == 0 or not np.isfinite(sd_all):
        return np.nan
    inv_m, inv_n, inv_sd = 1.0 / m, 1.0 / n, 1.0 / sd_all
    total = d_all.sum()
    extreme = denom = 0
    for _ in range(mc):
        idx = rng.choice(total_count, size=m, replace=False)
        s = (((d_all[idx].sum() * inv_m) - ((total - d_all[idx].sum()) * inv_n)) * inv_sd)
        if np.isfinite(s):
            denom += 1
            if abs(s) >= abs(s_obs):
                extreme += 1
    return (extreme + 1) / (denom + 1) if denom > 0 else np.nan
