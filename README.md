# POLAR YouTube Channel Embeddings

This repository contains scripts, example configs, lightweight run outputs, and
plotting code for POLAR-style analysis of YouTube channel embeddings across the
Iran and Afghanistan war corpora.

## Repository Layout

- `scripts/` contains the training, scoring, and plotting entry points.
- `config/` contains sanitized example configs.
- `resources/` contains example attribute, anchor, sentiment, and topic lexicons.
- `outputs/` contains lightweight run metadata and score summaries.
- `notebooks/` contains the analysis notebook with outputs cleared.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Recreate `embedding_map.pdf`

The standalone script below recreates the notebook figure:

```bash
python scripts/plot_embedding_map.py
```

By default it reads:

- `outputs/iran/polar_youtube_run`
- `outputs/afghanistan/polar_youtube_run`

The lightweight metadata is included, but `model.safetensors` files are ignored
by git because each trained model is about 438 MB. Copy or regenerate those
artifacts before running the plot script, or point the script at existing run
directories:

```bash
python scripts/plot_embedding_map.py \
  --iran-run-dir /path/to/outputs/iran/polar_youtube_run \
  --afghanistan-run-dir /path/to/outputs/afghanistan/polar_youtube_run
```

Use `--label-all --output notebooks/images/embedding_map_with_all_labels.pdf` to
render the labeled variant.

## Training And Scoring

Train channel embeddings:

```bash
python scripts/trainer_youtube_channels_polar.py \
  --config config/iran/train_config_youtube_channels.example.json
```

Run POLAR scoring:

```bash
python scripts/polar_youtube_channels.py \
  --config config/iran/polar_config_youtube_channels.example.json
```

Anchor and topic sentiment scoring use the corresponding config files in
`config/`.

## Notes

Raw data and trained model weights are not tracked. Keep them in `data/raw/` or
the appropriate `outputs/*/polar_youtube_run/model/` directory locally.
