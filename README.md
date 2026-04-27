# Hypertext YouTube War Discourse

This repository is the code and analysis companion for a paper on war discourse
on YouTube. The project compares YouTube discussions of two U.S.-linked
conflicts: the Afghanistan withdrawal in 2021 and the Iran conflict escalation
in 2026.

The study treats YouTube war discourse as a linked interaction space spanning
videos, channels, audiences, comments, and replies. It analyzes four corpora
that separate news organizations from political influencers, covering 340,383
comments from 212,968 unique commenters.

## Paper Overview

The analysis combines:

- YouTube Data API collection of channel uploads and video metadata.
- Channel-level discourse embeddings trained from YouTube comment corpora.
- WEAT-style semantic association tests adapted to channel embeddings.
- Topic, anchor, and sentiment association analyses.
- Classifier-based toxicity and hate-related proxy scores used in the paper's
  hostility analysis.

The results show that Afghanistan and Iran discussions form distinct discourse
spaces, while some ideologically different actors converge toward similar
patterns of conflict commentary. Semantic associations around geopolitical terms
vary across conflicts and source types, suggesting that the same vocabulary is
framed differently across communities. Hostility analysis further shows that
influencer-centered spaces tend to attract more toxic discussion, while the Iran
corpus exhibits stronger and more persistent hate-related signals, including in
replies.

## Repository Layout

- `collection/Channel_Data_collection.py` collects channel upload metadata from
  the YouTube Data API v3.
- `collection/countries.py` filters collected channel/video JSON by a title
  keyword and month window.
- `polar_analysis/` contains the channel embedding and semantic association
  pipeline.
- `polar_analysis/scripts/trainer_youtube_channels_polar.py` trains
  channel-level discourse embeddings from YouTube comments.
- `polar_analysis/scripts/polar_youtube_channels.py` runs WEAT-style semantic
  association scoring over trained channel embeddings.
- `polar_analysis/scripts/polar_youtube_channels_anchor.py` scores channels
  against anchor terms.
- `polar_analysis/scripts/polar_youtube_channels_topics.py` scores channels
  against topic bags.
- `polar_analysis/scripts/polar_youtube_channels_topic_sentiment.py` scores
  topic-sentiment associations.
- `polar_analysis/scripts/plot_embedding_map.py` recreates the embedding
  projection figure.
- `polar_analysis/config/` contains sanitized example configs for Afghanistan
  and Iran runs.
- `polar_analysis/resources/` contains example attribute, anchor, sentiment, and
  topic lexicons.
- `polar_analysis/outputs/` contains lightweight metadata and score summaries
  from the runs.
- `polar_analysis/notebooks/` contains the cleaned analysis notebook.
- `data/` is reserved for local raw data and is not tracked.

Raw comment exports, API keys, and trained model weights are not tracked.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`collection/Channel_Data_collection.py` also requires access to the YouTube Data
API v3. Edit the script's configuration block before running it:

- `API_KEY`: your YouTube Data API key.
- `OUTPUT_JSON`: where the collected payload should be written.
- `TARGET_MONTH_PREFIXES`: months to keep, such as `("2021-08", "2021-09")`.
- `STOP_BEFORE_MONTH`: earliest month boundary for stopping playlist traversal.
- `CHANNELS`: list of channel dictionaries with `name` and `url`.

Example channel entry:

```python
CHANNELS = [
    {"name": "Example Channel", "url": "https://www.youtube.com/@example"},
]
```

Run collection from the repository root:

```bash
python collection/Channel_Data_collection.py
```

The collector writes incrementally, so partial results are preserved if a
request fails or a long collection job is interrupted.

After collection, `collection/countries.py` can be used to create a smaller
video set for a conflict keyword and date window. Edit `INPUT_JSON`,
`OUTPUT_JSON`, `KEYWORD`, and `VALID_MONTH_PREFIXES` in that script before
running it.

## Training Channel Embeddings

Training configs expect raw YouTube comment exports under `data/raw/`. The
example paths are sanitized placeholders and should be adapted to the local data
location.

Run the training and scoring commands from `polar_analysis/`:

```bash
cd polar_analysis
```

Train the Iran model:

```bash
python scripts/trainer_youtube_channels_polar.py \
  --config config/iran/train_config_youtube_channels.example.json
```

Train the Afghanistan model:

```bash
python scripts/trainer_youtube_channels_polar.py \
  --config config/afghanistan/train_config_youtube_channels.example.json
```

Each training run writes channel metadata, tokenizer files, model files, and
run summaries into `polar_analysis/outputs/<corpus>/polar_youtube_run/`.

## Semantic Association Analyses

Run WEAT-style POLAR scoring:

```bash
python scripts/polar_youtube_channels.py \
  --config config/iran/polar_config_youtube_channels.example.json
```

Run anchor scoring:

```bash
python scripts/polar_youtube_channels_anchor.py \
  --config config/iran/anchor_config_youtube_channels.example.json
```

Run topic sentiment scoring:

```bash
python scripts/polar_youtube_channels_topic_sentiment.py \
  --config config/iran/topic_sentiment_config_youtube_channels.example.json
```

Use the corresponding files in `config/afghanistan/` for Afghanistan runs.

## Recreate `embedding_map.pdf`

The standalone plotting script recreates the paper's channel embedding map:

```bash
python polar_analysis/scripts/plot_embedding_map.py
```

By default it reads:

- `polar_analysis/outputs/iran/polar_youtube_run`
- `polar_analysis/outputs/afghanistan/polar_youtube_run`

The lightweight metadata and score files are included, but `model.safetensors`
files are ignored because each trained model is about 438 MB. Copy or regenerate
those artifacts before running the plot script, or point the script at existing
run directories:

```bash
python polar_analysis/scripts/plot_embedding_map.py \
  --iran-run-dir /path/to/outputs/iran/polar_youtube_run \
  --afghanistan-run-dir /path/to/outputs/afghanistan/polar_youtube_run
```

Use `--label-all --output polar_analysis/notebooks/images/embedding_map_with_all_labels.pdf`
from the repository root, or `--label-all --output notebooks/images/embedding_map_with_all_labels.pdf`
from `polar_analysis/`, to render the labeled variant.

## Data And Artifact Policy

The repository intentionally excludes:

- YouTube API keys.
- Raw comment exports.
- Large trained model weights.
- Generated PDF figures.

Keep raw data in `data/raw/` and trained model weights in the relevant
`polar_analysis/outputs/*/polar_youtube_run/model/` directory locally.
