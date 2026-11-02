# LFM / Mimi TTS

Run commands from `/home/speech/Documents/workspace/llm_tts`.
Python environment: `/data/util/anaconda3/envs/duplexomni/bin/python`.

- `scripts/`: training, inference, extraction and interactive entry points.
- `tts_pipeline/`: model, loss, text encoder and dataset modules.
- `data/`: generated manifests and preprocessing reports; source audio stays in `/data/db/LibriTTS_R`.
- `docs/`: detailed notes.
- `LFM3-1.7B/`: local pretrained model files.
- `checkpoints/`: created by training when selected as the output directory.

```bash
python -m scripts.export_libritts_json --output data/librispeech_data.json
CUDA_VISIBLE_DEVICES=0,1 /data/util/anaconda3/envs/duplexomni/bin/torchrun --standalone --nproc_per_node=2 -m scripts.train_depth_tts --online --manifest data/librispeech_data.json --output-dir checkpoints/lfm_mimi --valid-ratio 0.01 --seed 42 --max-audio-seconds 10 --batch-size 2 --gradient-accumulation 4 --epochs 20
python -m scripts.infer_depth_tts --checkpoint checkpoints/lfm_mimi/last.pt --text "Hello."
```

Training automatically splits utterances into 99% train / 1% validation using a
fixed seed. Speakers may overlap between splits. `split.json` records the source
and indices (relative to the smoke subset when `--smoke-samples` is used).
Keep the manifest order unchanged to reproduce a split. Validation runs every
epoch without gradients; loss is averaged over valid tokens per codebook, then
over codebooks. `last.pt` stores the latest epoch and `best.pt` the lowest validation
loss. Use a fresh output directory for each run. `--valid-ratio 0` disables validation.
For distributed training, `--batch-size` is per GPU. With two GPUs, batch size 2 and
gradient accumulation 4, the effective batch size is 16 utterances per optimizer step.
TensorBoard logs are written under `output-dir/tensorboard` by default. The latest
checkpoint is `last.pt`, the best is `best.pt`, and `--save-top-k 3` additionally keeps
the three checkpoints with the lowest validation loss as `best_epoch_*.pt`.
The console and TensorBoard also report overall and per-codebook token accuracy;
padding targets are excluded, while EOS is included as a valid codebook-0 target.
Training code tensors use the longest sequence in each batch, up to 2048 frames by
default. Shorter items are padded with ignored targets (`-100`). Samples that exceed
the limit are truncated to 2047 audio frames and receive EOS at frame 2047. Override
the limit with `--max-frames`.
For a duration-based limit, use `--max-audio-seconds 10`. Online manifests are
filtered to complete utterances no longer than 10 seconds, avoiding transcript/audio
mismatch. At Mimi's 12.5 Hz this creates at most 125 audio frames plus one EOS slot.

The frozen LFM extracts text hidden states once. These are padded to the acoustic
length and passed directly to the depth decoder. There is no second LFM forward
or previous-frame conditioning. Attention is across codebooks within each frame;
PAD-only frames do not have temporal context. Earlier checkpoints containing a
temporal backbone are incompatible. The older architecture description in
`docs/README_TTS.md` is historical.
