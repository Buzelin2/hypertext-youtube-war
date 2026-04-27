#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from dataclasses import dataclass, field
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
DEFAULT_TOPICS_PATH = PROJECT_DIR / "resources" / "topics" / "topics_war_youtube.example.json"
DEFAULT_ATTRIBUTES_PATH = PROJECT_DIR / "resources" / "attributes" / "sentiment_words.example.json"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "topic_sentiment_config_youtube_channels.example.json"


@dataclass
class Cfg:
    out_dir: str = str(DEFAULT_OUTPUT_DIR)
    model_dir: Optional[str] = None
    users_csv: Optional[str] = None
    users_json: Optional[str] = None
    meta_json: Optional[str] = None
    topics_path: str = str(DEFAULT_TOPICS_PATH)
    attributes_path: str = str(DEFAULT_ATTRIBUTES_PATH)
    attribute_pair: str = "sentiment"
    phrase_templates: List[str] = field(default_factory=lambda: ["{attr} {term}", "{term} is {attr}"])
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
        for key in ("out_dir", "model_dir", "users_csv", "users_json", "meta_json", "topics_path", "attributes_path"):
            if key in data:
                data[key] = resolve_config_path(str(config_path), data[key])
        return cls(**data)


def _parse_topic_terms(raw: Any) -> Tuple[List[str], Dict[str, Any]]:
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
        raise TypeError(f"Topic spec must be a string, list, or object, got {type(raw).__name__}.")
    cleaned = [t.strip() for t in terms if isinstance(t, str) and t.strip()]
    return cleaned, meta


def _build_phrases(terms: List[str], attrs: List[str], templates: List[str]) -> List[str]:
    phrases: List[str] = []
    for term in terms:
        for attr in attrs:
            for template in templates:
                phrase = template.format(term=term, attr=attr).strip()
                if phrase:
                    phrases.append(phrase)
    # preserve order, dedupe
    return list(dict.fromkeys(phrases))


def run(cfg: Cfg) -> None:
    np.random.seed(cfg.seed)

    model_dir, users_csv, users_json, meta_json = resolve_model_and_user_paths(
        cfg.out_dir, cfg.model_dir, cfg.users_csv, cfg.users_json, cfg.meta_json
    )
    users = load_users(users_csv, users_json, cfg.min_posts_default, meta_json, cfg.usr_prefix_fallback)
    if not users:
        raise SystemExit("No users after filtering.")

    topics_raw = json.loads(Path(cfg.topics_path).read_text(encoding="utf-8"))
    if not isinstance(topics_raw, dict):
        raise TypeError("Topics JSON must be an object keyed by topic name.")

    attrs_raw = json.loads(Path(cfg.attributes_path).read_text(encoding="utf-8"))
    if not isinstance(attrs_raw, dict):
        raise TypeError("Attributes JSON must be an object keyed by pair name.")
    pair = attrs_raw.get(cfg.attribute_pair)
    if not isinstance(pair, dict):
        raise KeyError(f"Attribute pair '{cfg.attribute_pair}' not found in {cfg.attributes_path}.")

    pos_attrs = [x for x in (pair.get("pos") or []) if isinstance(x, str) and x.strip()]
    neg_attrs = [x for x in (pair.get("neg") or []) if isinstance(x, str) and x.strip()]
    if not pos_attrs or not neg_attrs:
        raise SystemExit("Selected attribute pair must contain non-empty pos and neg lists.")

    print(f"users={len(users)}")

    tok, embeddings, vocab = load_tokenizer_and_embeddings(model_dir)
    cache: Dict[str, Optional[np.ndarray]] = {}

    topic_mats: Dict[str, Tuple[np.ndarray, np.ndarray, int, int, Dict[str, Any], List[str], List[str]]] = {}
    diagnostics: List[Dict[str, Any]] = []
    for topic_name, spec in topics_raw.items():
        terms, extra_meta = _parse_topic_terms(spec)
        pos_phrases = _build_phrases(terms, pos_attrs, cfg.phrase_templates)
        neg_phrases = _build_phrases(terms, neg_attrs, cfg.phrase_templates)

        A = [v for phrase in pos_phrases if (v := word_vec(phrase, tok, embeddings, cache)) is not None]
        B = [v for phrase in neg_phrases if (v := word_vec(phrase, tok, embeddings, cache)) is not None]
        A = np.vstack(A) if A else np.empty((0, embeddings.shape[1]), "float32")
        B = np.vstack(B) if B else np.empty((0, embeddings.shape[1]), "float32")
        sep = float(np.dot(l2n(A.mean(0)), l2n(B.mean(0)))) if len(A) > 0 and len(B) > 0 else np.nan
        topic_mats[topic_name] = (A, B, len(A), len(B), extra_meta, pos_phrases, neg_phrases)

        diagnostics.append(
            {
                "topic": topic_name,
                "label": extra_meta.get("label"),
                "description": extra_meta.get("description"),
                "group": extra_meta.get("group"),
                "topic_terms": terms,
                "n_pos_phrases": len(A),
                "n_neg_phrases": len(B),
                "phrase_templates": cfg.phrase_templates,
                "centroid_cosine": None if not np.isfinite(sep) else float(sep),
                "pos_phrases_sample": pos_phrases[:10],
                "neg_phrases_sample": neg_phrases[:10],
            }
        )
        print(f"[{topic_name}] pos={len(A)} neg={len(B)}")

    user_vecs, missing = build_user_vectors(users, vocab, embeddings)
    if not user_vecs:
        raise SystemExit("No user tokens found in the saved model vocabulary.")
    if missing:
        print(f"Missing {len(missing)} user tokens (showing up to 5): {missing[:5]}")

    rows: List[Dict[str, Any]] = []
    uids = list(user_vecs.keys())
    for tidx, (topic_name, payload) in enumerate(topic_mats.items(), 1):
        A, B, m, n, extra_meta, pos_phrases, neg_phrases = payload
        if m == 0 or n == 0:
            continue
        AB = np.vstack([A, B])
        for k, uid in enumerate(uids, 1):
            meta = users[uid]
            u = user_vecs[uid]
            sims = AB @ u
            sd = float(np.std(sims))
            score = (sims[:m].mean() - sims[m:].mean()) / sd if sd > 0 and np.isfinite(sd) else np.nan
            rows.append(
                {
                    "user_id": uid,
                    "topic": topic_name,
                    "topic_label": extra_meta.get("label") or topic_name,
                    "topic_group": extra_meta.get("group"),
                    "topic_description": extra_meta.get("description"),
                    "topic_sentiment_score": None if not np.isfinite(score) else float(score),
                    "topic_sentiment_direction": "positive" if np.isfinite(score) and score > 0 else ("negative" if np.isfinite(score) and score < 0 else "neutral"),
                    "n_pos_phrases": m,
                    "n_neg_phrases": n,
                    "pair_centroid_cosine": None,
                    "n_posts": int(meta.get("n_posts", 0)),
                    "channel_id": meta.get("channel_id"),
                    "channel_title": meta.get("channel_title"),
                    "group_label": meta.get("group_label"),
                    "corpus": meta.get("corpus"),
                    "source_kind": meta.get("source_kind"),
                    "n_videos": meta.get("n_videos"),
                }
            )
            if k % cfg.print_every == 0:
                print(f"topic {tidx}/{len(topic_mats)} {k}/{len(uids)} users")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["topic_sentiment_zscore_within_topic"] = (
            df.groupby("topic", sort=False)["topic_sentiment_score"]
            .transform(lambda s: (s - s.mean()) / s.std(ddof=0) if float(s.std(ddof=0)) > 0 else 0.0)
        )
    else:
        df["topic_sentiment_zscore_within_topic"] = []

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_user_topic_sentiment_scores.json").write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if cfg.save_csv:
        df.to_csv(out_dir / "per_user_topic_sentiment_scores.csv", index=False)
    (out_dir / "topic_sentiment_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summaries = []
    for topic_name, sub in df.groupby("topic", sort=False):
        vals = pd.to_numeric(sub["topic_sentiment_score"], errors="coerce")
        summaries.append(
            {
                "topic": topic_name,
                "n_users": int(sub.shape[0]),
                "mean_topic_sentiment_score": None if vals.dropna().empty else float(vals.mean()),
                "std_topic_sentiment_score": None if vals.dropna().empty else float(vals.std(ddof=0)),
                "min_topic_sentiment_score": None if vals.dropna().empty else float(vals.min()),
                "max_topic_sentiment_score": None if vals.dropna().empty else float(vals.max()),
            }
        )
    (out_dir / "topic_sentiment_summaries.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    run_summary = {
        "out_dir": str(out_dir),
        "model_dir": str(model_dir),
        "topics_path": cfg.topics_path,
        "attributes_path": cfg.attributes_path,
        "attribute_pair": cfg.attribute_pair,
        "phrase_templates": cfg.phrase_templates,
        "n_users_requested": len(users),
        "n_users_scored": len(user_vecs),
        "n_topics_requested": len(topics_raw),
        "n_topics_scored": len(topic_mats),
        "n_missing_user_tokens": len(missing),
        "missing_user_tokens": missing,
        "files": {
            "per_user_topic_sentiment_scores_json": str(out_dir / "per_user_topic_sentiment_scores.json"),
            "per_user_topic_sentiment_scores_csv": str(out_dir / "per_user_topic_sentiment_scores.csv") if cfg.save_csv else None,
            "topic_sentiment_summaries_json": str(out_dir / "topic_sentiment_summaries.json"),
            "topic_sentiment_diagnostics_json": str(out_dir / "topic_sentiment_diagnostics.json"),
        },
    }
    (out_dir / "topic_sentiment_run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score channel sentiment toward topic bags using positive and negative phrase compositions.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    run(Cfg.from_json(args.config))


if __name__ == "__main__":
    main()
