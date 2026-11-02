# LFM hidden state → LLMVoX-style Mimi depth TTS

This pipeline keeps text raw, extracts frozen LFM assistant hidden states in the
training step, encodes paired 24 kHz speech into 4 Mimi codebooks online, and
trains an LLMVoX-style causal temporal decoder plus a codebook depth decoder.

## 1. Input manifest

Create `data/train.jsonl` with one utterance per line:

```json
{"id":"000001","audio":"/absolute/path/000001.wav","text":"안녕하세요."}
```

Audio may have any sample rate and may be stereo. Preprocessing converts it to
mono 24 kHz. Transcripts must already be normalized exactly as they are spoken.

## 2. Precompute features

```bash
python -m scripts.prepare_tts_data \
  --manifest data/train.jsonl \
  --output-dir data/processed \
  --lfm-model ./LFM3-1.7B \
  --num-quantizers 4
```

Each `.pt` contains `text_hidden [text_tokens, 2048]` and
`audio_codes [4, audio_frames]`. Mimi runs at 12.5 frames/second.

## 3. Train

```bash
python -m scripts.train_depth_tts \
  --manifest data/processed/manifest.jsonl \
  --output-dir checkpoints/depth_tts \
  --batch-size 2 \
  --epochs 100
```

To keep text raw and extract only Mimi codec tokens online from the raw manifest:

```bash
python -m scripts.train_depth_tts \
  --online \
  --manifest data/train.jsonl \
  --lfm-model ./LFM3-1.7B \
  --num-quantizers 4 \
  --output-dir checkpoints/depth_tts \
  --batch-size 2 \
  --epochs 100
```

The DataLoader returns raw text strings and extracts only Mimi codes. The training
step then passes those strings through the frozen LFM to obtain hidden states.
Online mode deliberately uses `num_workers=0`: CUDA-backed models must stay in
the training process and must not be copied into worker processes.

The default decoder follows the public LLMVoX configuration: 768 dimensions,
4 pre-norm causal GPT layers, 8 attention heads, a 4x GELU MLP, learned position
embeddings, no linear bias, and zero dropout. At each audio frame it concatenates
the projected LFM condition with the previous Mimi-frame embedding and applies
L2 normalization. Text positions after the last text token use a learned PAD
embedding. Codebook 0 has one extra `EOS=2048` token; codebooks 1–3 retain the
native Mimi vocabulary `0..2047`. During training EOS is appended immediately
after the final audio frame only on codebook 0.

The depth decoder uses the same LLMVoX GPT block over the 4 Mimi codebooks:

```text
temporal state -> codebook 0
+ codebook 0   -> codebook 1
+ codebook 1   -> codebook 2
+ codebook 2   -> codebook 3
```

## 4. Synthesize

```bash
python -m scripts.infer_depth_tts \
  --checkpoint checkpoints/depth_tts/last.pt \
  --text "안녕하세요." \
  --output output.wav
```

Inference runs causally until codebook 0 predicts EOS. EOS itself and its depth
codes are not sent to Mimi. Use `--max-frames` as a hard safety limit. Mimi
decodes the generated `[1, 4, frames]` codes to 24 kHz audio.
