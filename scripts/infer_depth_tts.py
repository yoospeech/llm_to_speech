"""Inference entry point: python -m scripts.infer_depth_tts."""
import argparse

import torch
import torchaudio
from transformers import MimiModel

from tts_pipeline.streaming_depth_tts import StreamingDepthTTS, StreamingDepthTTSConfig
from tts_pipeline.text_embedder import LFMTextEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--lfm-model", default="./LFM3-1.7B")
    parser.add_argument("--mimi-model", default="kyutai/mimi")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--min-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = StreamingDepthTTSConfig(**checkpoint["config"])
    embedder = LFMTextEmbedder(args.lfm_model, str(device))
    tts = StreamingDepthTTS(config).to(device)
    tts.load_state_dict(checkpoint["model"])
    tts.eval()

    hidden, _ = embedder(args.text)
    hidden = hidden.unsqueeze(0).to(device)
    text_mask = torch.ones(hidden.shape[:2], device=device, dtype=torch.bool)
    codes = tts.generate(
        hidden,
        text_mask,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        temperature=args.temperature,
    )

    mimi = MimiModel.from_pretrained(args.mimi_model).to(device).eval()
    with torch.inference_mode():
        audio = mimi.decode(codes).audio_values
    samples = audio[0, 0].float().cpu()
    torchaudio.save(args.output, samples.unsqueeze(0), 24000)
    print(f"saved {args.output} ({codes.shape[-1]} Mimi frames)")


if __name__ == "__main__":
    main()
