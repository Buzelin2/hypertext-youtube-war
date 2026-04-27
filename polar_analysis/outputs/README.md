# Output Artifacts

This directory stores run outputs produced by the training and scoring scripts.

The lightweight JSON/CSV summaries can be tracked for reproducibility. Trained
model weights such as `model.safetensors` are intentionally ignored because they
are hundreds of megabytes per run and exceed normal GitHub file limits.

To recreate `polar_analysis/notebooks/images/embedding_map.pdf`, provide the
trained Hugging Face model files for these runs:

- `polar_analysis/outputs/iran/polar_youtube_run/model/model.safetensors`
- `polar_analysis/outputs/afghanistan/polar_youtube_run/model/model.safetensors`

Then run:

```bash
python polar_analysis/scripts/plot_embedding_map.py
```
