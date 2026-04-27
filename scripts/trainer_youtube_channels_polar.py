#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import random
import re
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AddedToken, AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "polar_youtube_run"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "train_config_youtube_channels.example.json"


@dataclass
class DatasetSpec:
    path: str
    group_label: Optional[str] = None
    corpus: Optional[str] = None
    source_kind: Optional[str] = None


@dataclass
class Cfg:
    datasets: List[DatasetSpec] = field(default_factory=list)
    out_dir: str = str(DEFAULT_OUTPUT_DIR)
    base_model: str = "bert-base-uncased"
    usr_prefix: str = "usr"
    max_len: int = 128
    epochs: int = 4
    batch_size: int = 128
    users_per_batch: int = 128
    lr: float = 5e-5
    mlm_prob: float = 0.15
    p_user_mask: float = 0.30
    seed: int = 42
    num_workers: int = max(2, (os.cpu_count() or 8) // 2)
    tokenize_chunk: int = 4096
    min_posts_per_user: int = 2
    export_kv: bool = False
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    warmup_ratio: float = 0.03
    per_user_cap: int = 200
    freeze_epochs: int = 1
    max_comments_per_user: Optional[int] = None
    align_use_hidden: bool = False
    align_lambda: float = 0.2
    con_weight: float = 0.0
    con_temperature: float = 0.07
    soft_prompt_len: int = 0
    include_top_level_comments: bool = True
    include_replies: bool = True
    min_comment_chars: int = 3
    deduplicate_within_user: bool = False
    merge_same_channel_across_datasets: bool = False

    @classmethod
    def from_json(cls, path: str) -> "Cfg":
        config_path = Path(path).expanduser().resolve()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"Config JSON must be an object, got {type(data).__name__}.")

        if "videos_with_comment_data" in data and "datasets" not in data:
            return cls(datasets=[DatasetSpec(path=str(config_path))])

        def _resolve_str_path(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = (config_path.parent / candidate).resolve()
            return str(candidate)

        if "datasets" in data:
            datasets_raw = data.pop("datasets") or []
        elif data.get("source_file"):
            datasets_raw = [{"path": data.pop("source_file")}]
        else:
            datasets_raw = []

        ds_fields = {f.name for f in fields(DatasetSpec)}
        datasets = []
        for item in datasets_raw:
            clean = {k: v for k, v in item.items() if k in ds_fields}
            if clean.get("path"):
                clean["path"] = _resolve_str_path(clean["path"])
            datasets.append(DatasetSpec(**clean))

        if "out_dir" in data:
            data["out_dir"] = _resolve_str_path(data["out_dir"])

        cfg_fields = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in data.items() if k in cfg_fields})
        cfg.datasets = datasets
        return cfg


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def l2n_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[(n == 0) | (~np.isfinite(n))] = 1.0
    return M / n


def uid_to_token(uid: str, tok, prefix: str = "usr") -> str:
    h = hashlib.sha1(str(uid).encode("utf-8")).hexdigest()[:10]
    token = f"{prefix}{h}"
    return token.lower() if getattr(tok, "do_lower_case", False) else token


def _clean_text(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None
    text = text.replace("\u200b", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _iter_comment_texts(thread: Dict[str, Any], include_top: bool, include_replies: bool) -> Iterable[str]:
    if include_top:
        top = (thread or {}).get("top_level_comment") or {}
        text = _clean_text(top.get("text_original") or top.get("text_display"))
        if text:
            yield text
    if include_replies:
        for reply in (thread or {}).get("replies") or []:
            text = _clean_text(reply.get("text_original") or reply.get("text_display"))
            if text:
                yield text


def _channel_fields(video_obj: Dict[str, Any]) -> Tuple[str, str]:
    api_meta = video_obj.get("api_video_metadata") or {}
    snippet = api_meta.get("snippet") or {}
    channel_id = snippet.get("channelId") or video_obj.get("channel_id") or "UNKNOWN_CHANNEL_ID"
    channel_title = (
        snippet.get("channelTitle")
        or video_obj.get("channel_title_from_input")
        or video_obj.get("input_video_metadata", {}).get("channel_title")
        or str(channel_id)
    )
    return str(channel_id), str(channel_title)


def _dataset_tag(spec: DatasetSpec) -> Dict[str, str]:
    path = Path(spec.path)
    stem = path.stem
    group_label = spec.group_label or stem
    corpus = spec.corpus or ("iran" if "iran" in stem.lower() else ("afghanistan" if "afghan" in stem.lower() else "unknown"))
    source_kind = spec.source_kind or ("news" if "news" in stem.lower() else "influencer")
    return {
        "group_label": group_label,
        "corpus": corpus,
        "source_kind": source_kind,
        "dataset_path": str(path),
        "dataset_name": stem,
    }


def load_youtube_channels(cfg: Cfg) -> Tuple[List[Tuple[str, str]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if not cfg.datasets:
        raise SystemExit("config.datasets must contain at least one JSON path.")

    samples: List[Tuple[str, str]] = []
    users: Dict[str, Dict[str, Any]] = {}
    load_summary: Dict[str, Any] = {"datasets": [], "total_raw_comments": 0, "total_kept_comments": 0}

    for spec in cfg.datasets:
        path = Path(spec.path)
        if not path.is_file():
            raise SystemExit(f"Input JSON not found: {path}")
        tag = _dataset_tag(spec)
        payload = json.loads(path.read_text(encoding="utf-8"))
        videos = payload.get("videos_with_comment_data") or []
        if not isinstance(videos, list):
            raise SystemExit(f"Unexpected schema in {path}: missing videos_with_comment_data list")

        ds_stats = {
            **tag,
            "n_videos": len(videos),
            "raw_comments": 0,
            "kept_comments": 0,
            "n_channels_before_filter": 0,
            "n_channels_after_filter": 0,
        }

        per_user_seen = defaultdict(set) if cfg.deduplicate_within_user else None

        for video in videos:
            channel_id, channel_title = _channel_fields(video)
            user_id = channel_id if cfg.merge_same_channel_across_datasets else f"{tag['group_label']}::{channel_id}"

            meta = users.setdefault(
                user_id,
                {
                    "n_posts": 0,
                    "label_majority": tag["source_kind"],
                    "targets": [],
                    "channel_id": channel_id,
                    "channel_title": channel_title,
                    "group_label": tag["group_label"],
                    "corpus": tag["corpus"],
                    "source_kind": tag["source_kind"],
                    "dataset_paths": set(),
                    "dataset_names": set(),
                    "video_ids": set(),
                    "video_titles": [],
                    "n_videos": 0,
                    "comments_per_video": {},
                },
            )
            meta["dataset_paths"].add(tag["dataset_path"])
            meta["dataset_names"].add(tag["dataset_name"])

            video_id = str(video.get("video_id") or "")
            if video_id and video_id not in meta["video_ids"]:
                meta["video_ids"].add(video_id)
                meta["n_videos"] += 1
            title = _clean_text(
                (video.get("api_video_metadata") or {}).get("snippet", {}).get("title")
                or (video.get("input_video_metadata") or {}).get("title")
            )
            if title and len(meta["video_titles"]) < 200:
                meta["video_titles"].append(title)

            kept_for_video = 0
            for thread in video.get("comment_threads") or []:
                for text in _iter_comment_texts(thread, cfg.include_top_level_comments, cfg.include_replies):
                    ds_stats["raw_comments"] += 1
                    load_summary["total_raw_comments"] += 1
                    if len(text) < int(cfg.min_comment_chars):
                        continue
                    if per_user_seen is not None and text in per_user_seen[user_id]:
                        continue
                    if per_user_seen is not None:
                        per_user_seen[user_id].add(text)
                    samples.append((user_id, text))
                    meta["n_posts"] += 1
                    ds_stats["kept_comments"] += 1
                    load_summary["total_kept_comments"] += 1
                    kept_for_video += 1
                    if cfg.max_comments_per_user is not None and meta["n_posts"] >= int(cfg.max_comments_per_user):
                        break
                if cfg.max_comments_per_user is not None and meta["n_posts"] >= int(cfg.max_comments_per_user):
                    break
            if video_id:
                meta["comments_per_video"][video_id] = kept_for_video

        ds_stats["n_channels_before_filter"] = len(
            {(m["group_label"], m["channel_id"]) for m in users.values() if tag["dataset_path"] in m["dataset_paths"]}
        )
        load_summary["datasets"].append(ds_stats)

    filtered_users: Dict[str, Dict[str, Any]] = {}
    filtered_samples: List[Tuple[str, str]] = []
    allowed = {uid for uid, meta in users.items() if meta["n_posts"] >= max(1, int(cfg.min_posts_per_user))}

    for uid, text in samples:
        if uid in allowed:
            filtered_samples.append((uid, text))
    if not filtered_samples:
        raise SystemExit("No comments after filtering by min_posts_per_user.")

    for uid in allowed:
        meta = users[uid]
        filtered_users[uid] = {
            **meta,
            "targets": list(meta.get("targets") or []),
            "dataset_paths": sorted(meta["dataset_paths"]),
            "dataset_names": sorted(meta["dataset_names"]),
            "video_ids": sorted(meta["video_ids"]),
            "video_titles": meta["video_titles"],
        }

    for ds in load_summary["datasets"]:
        ds["n_channels_after_filter"] = sum(1 for meta in filtered_users.values() if ds["dataset_path"] in meta["dataset_paths"])

    load_summary["n_users_after_filter"] = len(filtered_users)
    load_summary["n_samples_after_filter"] = len(filtered_samples)
    return filtered_samples, filtered_users, load_summary


class EncodedDataset(Dataset):
    def __init__(self, enc: Dict[str, List[List[int]]]):
        self.enc = enc

    def __len__(self) -> int:
        return len(self.enc["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            key: torch.tensor(self.enc[key][idx], dtype=torch.long)
            for key in ("input_ids", "attention_mask", "special_tokens_mask")
        }


def batch_encode(tok, texts: List[str], max_len: int, chunk: int) -> Dict[str, List[List[int]]]:
    ids, attn, special = [], [], []
    for i in range(0, len(texts), chunk):
        enc = tok(
            texts[i : i + chunk],
            truncation=True,
            max_length=max_len,
            padding=False,
            return_special_tokens_mask=True,
        )
        ids.extend(enc["input_ids"])
        attn.extend(enc["attention_mask"])
        special.extend(enc["special_tokens_mask"])
    return {"input_ids": ids, "attention_mask": attn, "special_tokens_mask": special}


class BalancedUserBatchSampler(Sampler[List[int]]):
    def __init__(self, sample_uids: List[str], batch_size: int, users_per_batch: int, per_user_cap: int = 0, drop_last: bool = False):
        self.bs = int(batch_size)
        self.upb = int(max(1, users_per_batch))
        self.drop_last = drop_last
        self.cap = int(per_user_cap)
        self.s_per_user = max(1, self.bs // self.upb)

        bins = defaultdict(list)
        for idx, uid in enumerate(map(str, sample_uids)):
            bins[uid].append(idx)
        self.bins = {uid: np.array(indexes, dtype=int) for uid, indexes in bins.items()}
        self.n_samples = len(sample_uids)

    def __iter__(self):
        rng = np.random.default_rng()
        local = {
            uid: deque(
                rng.permutation(indexes)[: self.cap].tolist() if self.cap > 0 and len(indexes) > self.cap else rng.permutation(indexes).tolist()
            )
            for uid, indexes in self.bins.items()
        }
        pool = [uid for uid, items in local.items() if items]

        while pool:
            batch, emptied = [], []
            take = min(self.upb, len(pool))
            for uid in rng.choice(pool, size=take, replace=False):
                for _ in range(self.s_per_user):
                    if local[uid]:
                        batch.append(local[uid].popleft())
                    else:
                        break
                if not local[uid]:
                    emptied.append(uid)
                if len(batch) >= self.bs:
                    break

            pool = [uid for uid in pool if uid not in emptied]
            if not batch:
                break
            if len(batch) < self.bs and self.drop_last:
                break
            yield batch[: self.bs]

    def __len__(self) -> int:
        return max(1, math.ceil(self.n_samples / max(1, self.bs)))


class UserAwareMLMCollator:
    def __init__(self, tok, mlm_prob: float = 0.15, p_user_mask: float = 0.30, user_token_ids: Optional[torch.Tensor] = None):
        self.tok = tok
        self.mlm = float(mlm_prob)
        self.pusr = float(p_user_mask)
        self.user_ids = user_token_ids if isinstance(user_token_ids, torch.Tensor) else torch.tensor([], dtype=torch.long)
        self.never_ids = [tok.cls_token_id, tok.sep_token_id, tok.pad_token_id, tok.unk_token_id, tok.mask_token_id]
        self.never_ids = [idx for idx in self.never_ids if idx is not None]

    def _mask(self, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        device = ids.device
        never = torch.isin(ids, torch.tensor(self.never_ids, device=device)) if self.never_ids else torch.zeros_like(ids, dtype=torch.bool)
        is_user = torch.isin(ids, self.user_ids.to(device)) if len(self.user_ids) > 0 else torch.zeros_like(ids, dtype=torch.bool)
        allowed_nonuser = attn & (~never) & (~is_user)
        allowed_user = attn & (~never) & is_user
        mask = (torch.rand_like(ids, dtype=torch.float) < self.mlm) & allowed_nonuser
        if len(self.user_ids) > 0:
            mask |= (torch.rand_like(ids, dtype=torch.float) < self.pusr) & allowed_user
        for b in range(ids.size(0)):
            if not mask[b].any():
                allowed = (allowed_nonuser[b] | allowed_user[b]).nonzero(as_tuple=False).flatten()
                if allowed.numel() > 0:
                    j = torch.randint(0, allowed.numel(), (1,), device=device)
                    mask[b, allowed[j]] = True
        return mask

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = self.tok.pad(features, padding=True, return_tensors="pt")
        ids = batch["input_ids"]
        attn = batch["attention_mask"].bool()
        user_first_ids = ids[:, 0].clone()

        labels = ids.clone()
        mask = self._mask(ids, attn)
        labels[~mask] = -100

        rand = torch.rand_like(ids, dtype=torch.float)
        mask80 = mask & (rand < 0.8)
        mask10 = mask & (rand >= 0.8) & (rand < 0.9)
        ids[mask80] = self.tok.mask_token_id

        user_set = set(self.user_ids.tolist())
        never_set = set(self.never_ids)
        safe = torch.tensor([idx for idx in range(len(self.tok)) if idx not in user_set and idx not in never_set], device=ids.device, dtype=torch.long)
        if mask10.any():
            rand_ids = safe[torch.randint(0, safe.numel(), ids.shape, device=ids.device)]
            ids[mask10] = rand_ids[mask10]

        return {"input_ids": ids, "attention_mask": attn.long(), "labels": labels, "user_first_ids": user_first_ids}


class SoftPromptTable(nn.Module):
    def __init__(self, num_users: int, dim: int, prompt_len: int):
        super().__init__()
        self.emb = nn.Embedding(num_users, prompt_len * dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        self.dim = dim
        self.prompt_len = prompt_len

    def forward(self, row_idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(row_idx)
        return x.view(x.size(0), self.prompt_len, self.dim)


def export_user_kv(model, tokenizer, usr_tokens_map: Dict[str, str], out_path: Path) -> None:
    try:
        from gensim.models import KeyedVectors
    except Exception:
        print("gensim not found; skipping KV export")
        return

    W = model.get_input_embeddings().weight.detach().cpu().numpy().astype("float32")
    W = l2n_rows(W)
    vocab = tokenizer.get_vocab()
    rows = []
    for uid, tok_str in usr_tokens_map.items():
        idx = vocab.get(tok_str)
        if idx is not None:
            rows.append((f"USR:{uid}", W[idx]))

    if not rows:
        print("No user tokens available for KV export.")
        return

    keys, vecs = zip(*rows)
    kv = KeyedVectors(vector_size=W.shape[1])
    kv.add_vectors(list(keys), np.vstack(vecs))
    kv.fill_norms()
    kv.save(str(out_path))


def train(cfg: Cfg) -> None:
    set_seeds(cfg.seed)
    out_dir = Path(cfg.out_dir)
    (out_dir / "model").mkdir(parents=True, exist_ok=True)

    samples, users, load_summary = load_youtube_channels(cfg)
    print(f"samples={len(samples)} users={len(users)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)

    user_tokens_map = {uid: uid_to_token(uid, tok, prefix=cfg.usr_prefix) for uid in users}
    tok.add_tokens([AddedToken(t, single_word=True, normalized=True) for t in user_tokens_map.values()])

    texts = [f"{user_tokens_map[uid]} {text}" for uid, text in samples]
    sample_uids = [uid for uid, _ in samples]

    vocab = tok.get_vocab()
    user_ids = {t: vocab[t] for t in user_tokens_map.values() if t in vocab}
    assert len(user_ids) == len(user_tokens_map), "Some user tokens were not added to the vocabulary."

    def _first_is_user_token(text: str) -> bool:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return len(ids) > 0 and ids[0] in user_ids.values()

    check_n = min(1000, len(texts))
    ok = sum(_first_is_user_token(texts[i]) for i in range(check_n))
    assert ok == check_n, "User token is not the first token."

    enc = batch_encode(tok, texts, cfg.max_len, cfg.tokenize_chunk)
    ds = EncodedDataset(enc)
    user_tensor = torch.tensor([vocab[t] for t in user_tokens_map.values()], dtype=torch.long)
    collator = UserAwareMLMCollator(tok, cfg.mlm_prob, cfg.p_user_mask, user_tensor)

    users_per_batch = min(cfg.users_per_batch, max(1, len(users)))
    sampler = BalancedUserBatchSampler(sample_uids, cfg.batch_size, users_per_batch, per_user_cap=cfg.per_user_cap, drop_last=False)
    dl = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    model = AutoModelForMaskedLM.from_pretrained(cfg.base_model)
    model.resize_token_embeddings(len(tok))
    model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    emb_layer = model.get_input_embeddings()

    token_id_list = [vocab[user_tokens_map[uid]] for uid in users]
    tokid2row = {tid: idx for idx, tid in enumerate(token_id_list)}
    soft_prompt = None
    if cfg.soft_prompt_len > 0:
        soft_prompt = SoftPromptTable(len(token_id_list), emb_layer.embedding_dim, cfg.soft_prompt_len).to(device)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if any(key in name for key in ["bias", "LayerNorm.weight"]) else decay).append(param)
    if soft_prompt is not None:
        decay.append(soft_prompt.emb.weight)

    fused_ok = (device.type == "cuda") and ("fused" in inspect.signature(AdamW).parameters)
    opt = AdamW(
        [
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        eps=1e-6,
        betas=(0.9, 0.98),
        **({"fused": True} if fused_ok else {}),
    )

    use_bf16 = (device.type == "cuda") and torch.cuda.is_bf16_supported()
    use_fp16 = (device.type == "cuda") and not use_bf16
    scaler = GradScaler("cuda", enabled=use_fp16)

    total_steps = cfg.epochs * max(1, len(dl)) // max(1, cfg.grad_accum_steps)
    warmup = max(100, int(cfg.warmup_ratio * total_steps))
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)

    embed_param_ids = {id(p) for p in emb_layer.parameters()}

    def set_backbone_trainable(trainable: bool) -> None:
        for param in model.parameters():
            param.requires_grad = id(param) in embed_param_ids or trainable

    set_backbone_trainable(False)
    model.train()

    user_token_tensor = torch.tensor([vocab[t] for t in user_tokens_map.values()], device=device, dtype=torch.long)
    gstep, running, t0 = 0, 0.0, time.time()

    for epoch in range(cfg.epochs):
        if epoch >= cfg.freeze_epochs:
            set_backbone_trainable(True)

        for batch in dl:
            user_first_ids = batch.pop("user_first_ids").to(device)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            need_hid = bool(cfg.align_use_hidden or cfg.con_weight > 0.0)

            inputs = {"attention_mask": batch["attention_mask"], "labels": batch["labels"]}
            if soft_prompt is not None and cfg.soft_prompt_len > 0:
                row_idx = torch.tensor([tokid2row[int(x)] for x in user_first_ids.tolist()], device=device, dtype=torch.long)
                prefix = soft_prompt(row_idx)
                word_emb = emb_layer(batch["input_ids"])
                inputs_embeds = torch.cat([prefix, word_emb], dim=1)
                attn_prefix = torch.ones((inputs_embeds.size(0), cfg.soft_prompt_len), dtype=batch["attention_mask"].dtype, device=device)
                labels_prefix = torch.full((inputs_embeds.size(0), cfg.soft_prompt_len), -100, dtype=batch["labels"].dtype, device=device)
                inputs["inputs_embeds"] = inputs_embeds
                inputs["attention_mask"] = torch.cat([attn_prefix, batch["attention_mask"]], dim=1)
                inputs["labels"] = torch.cat([labels_prefix, batch["labels"]], dim=1)
            else:
                inputs = {**batch}

            if use_bf16:
                with autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inputs, output_hidden_states=need_hid)
                    loss = out.loss
            elif use_fp16:
                with autocast("cuda", dtype=torch.float16):
                    out = model(**inputs, output_hidden_states=need_hid)
                    loss = out.loss
            else:
                out = model(**inputs, output_hidden_states=need_hid)
                loss = out.loss

            ids = batch["input_ids"]
            attn = batch["attention_mask"].bool()
            never_ids = torch.tensor([tok.cls_token_id, tok.sep_token_id, tok.pad_token_id, tok.unk_token_id, tok.mask_token_id], device=ids.device)
            never_ids = never_ids[never_ids >= 0]
            is_never = torch.isin(ids, never_ids)
            is_user = torch.isin(ids, user_token_tensor)
            ctx_mask = attn & (~is_never) & (~is_user)

            e_usr = emb_layer(user_first_ids)
            if cfg.align_use_hidden and need_hid and out.hidden_states is not None:
                last_hid = out.hidden_states[-1]
                if soft_prompt is not None and cfg.soft_prompt_len > 0:
                    pad = torch.zeros((ids.size(0), cfg.soft_prompt_len), dtype=torch.bool, device=ids.device)
                    ctx_mask_ext = torch.cat([pad, ctx_mask], dim=1)
                    ctx = torch.where(ctx_mask_ext.unsqueeze(-1), last_hid, torch.zeros_like(last_hid))
                else:
                    ctx = torch.where(ctx_mask.unsqueeze(-1), last_hid, torch.zeros_like(last_hid))
            else:
                input_emb = emb_layer(ids)
                ctx = torch.where(ctx_mask.unsqueeze(-1), input_emb, torch.zeros_like(input_emb))

            ctx_sum = ctx.sum(dim=1)
            ctx_cnt = (ctx_mask.sum(dim=1).clamp_min(1)).unsqueeze(-1).to(ctx_sum.dtype)
            e_ctx_mean = ctx_sum / ctx_cnt
            cos = nn.functional.cosine_similarity(e_usr, e_ctx_mean.detach(), dim=-1, eps=1e-8)
            L_align = (1.0 - cos).mean()
            loss_total = loss + cfg.align_lambda * L_align

            if cfg.con_weight > 0.0:
                u = nn.functional.normalize(e_usr, dim=-1)
                c = nn.functional.normalize(e_ctx_mean, dim=-1)
                logits_uc = (u @ c.t()) / cfg.con_temperature
                logits_cu = (c @ u.t()) / cfg.con_temperature
                target = torch.arange(u.shape[0], device=ids.device)
                ce = nn.CrossEntropyLoss()
                L_con = 0.5 * (ce(logits_uc, target) + ce(logits_cu, target))
                loss_total = loss_total + cfg.con_weight * L_con
            else:
                L_con = torch.tensor(0.0, device=ids.device)

            if not torch.isfinite(loss_total):
                print("[warn] non-finite loss; skipping step")
                opt.zero_grad(set_to_none=True)
                continue

            if use_fp16:
                scaler.scale(loss_total / cfg.grad_accum_steps).backward()
                if (gstep + 1) % cfg.grad_accum_steps == 0:
                    if cfg.grad_clip > 0:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
            else:
                (loss_total / cfg.grad_accum_steps).backward()
                if (gstep + 1) % cfg.grad_accum_steps == 0:
                    if cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()

            gstep += 1
            running += float(loss.item())
            if gstep % 100 == 0:
                sps = gstep / max(1e-9, time.time() - t0)
                print(
                    f"epoch={epoch + 1} step={gstep} mlm={running / 100:.4f} "
                    f"align={L_align.detach().item():.4f} con={L_con.detach().item():.4f} {sps:.2f}sps"
                )
                running = 0.0

    model.save_pretrained(str(out_dir / "model"))
    tok.save_pretrained(str(out_dir / "model"))

    users_json_rows = []
    with (out_dir / "users.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "token", "n_posts", "label_majority", "targets", "channel_id", "channel_title", "group_label", "corpus", "source_kind", "n_videos"],
        )
        writer.writeheader()
        for uid, meta in sorted(users.items()):
            row = {
                "user_id": uid,
                "token": user_tokens_map[uid],
                "n_posts": meta.get("n_posts", 1),
                "label_majority": meta.get("label_majority"),
                "targets": ",".join(meta.get("targets") or []),
                "channel_id": meta.get("channel_id"),
                "channel_title": meta.get("channel_title"),
                "group_label": meta.get("group_label"),
                "corpus": meta.get("corpus"),
                "source_kind": meta.get("source_kind"),
                "n_videos": meta.get("n_videos", 0),
            }
            writer.writerow(row)
            users_json_rows.append(
                {
                    **row,
                    "dataset_paths": meta.get("dataset_paths", []),
                    "dataset_names": meta.get("dataset_names", []),
                    "video_ids": meta.get("video_ids", []),
                    "video_titles": meta.get("video_titles", []),
                    "comments_per_video": meta.get("comments_per_video", {}),
                }
            )

    (out_dir / "users.json").write_text(json.dumps(users_json_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        **asdict(cfg),
        "datasets": [asdict(x) for x in cfg.datasets],
        "n_users": len(users),
        "n_texts": len(texts),
        "loader_summary": load_summary,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    training_summary = {
        "out_dir": str(out_dir),
        "model_dir": str(out_dir / "model"),
        "n_users": len(users),
        "n_samples": len(samples),
        "base_model": cfg.base_model,
        "device": str(device),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "users_per_batch": users_per_batch,
        "seed": cfg.seed,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(training_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.export_kv:
        export_user_kv(model, tok, user_tokens_map, out_dir / "user_embeddings.kv")
    print(f"Saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train POLAR with each YouTube channel treated as a user.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to a JSON config file. Defaults to {DEFAULT_CONFIG_PATH}",
    )
    args = parser.parse_args()
    train(Cfg.from_json(args.config))


if __name__ == "__main__":
    main()
