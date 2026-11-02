"""Offline extraction entry point: python -m scripts.prepare_tts_data."""
import argparse
import json
from pathlib import Path

import torch
import torchaudio
from transformers import AutoFeatureExtractor, MimiModel

from tts_pipeline.text_embedder import LFMTextEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paired LFM-hidden/Mimi-code samples")
    parser.add_argument("--manifest", required=True, help='JSONL with {"audio": ..., "text": ...}')
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lfm-model", default="./LFM3-1.7B")
    parser.add_argument("--mimi-model", default="kyutai/mimi")
    parser.add_argument("--num-quantizers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_audio(path: str, sampling_rate: int) -> torch.Tensor:
    waveform, source_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if source_rate != sampling_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sampling_rate)
    return waveform.clamp(-1.0, 1.0)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest_dir = manifest_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    text_embedder = LFMTextEmbedder(args.lfm_model, args.device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.mimi_model)
    mimi = MimiModel.from_pretrained(args.mimi_model).to(device).eval()
    processed_manifest = output_dir / "manifest.jsonl"

    with manifest_path.open(encoding="utf-8") as source, processed_manifest.open(
        "w", encoding="utf-8"
    ) as destination:
        for index, line in enumerate(source):
            item = json.loads(line)
            text = item["text"]
            audio_path = Path(item["audio"])
            if not audio_path.is_absolute():
                audio_path = manifest_dir / audio_path
            audio_path = str(audio_path.resolve())
            sample_id = str(item.get("id", f"sample_{index:08d}"))

            text_hidden, text_token_ids = text_embedder(text)
            waveform = load_audio(audio_path, feature_extractor.sampling_rate)
            mimi_inputs = feature_extractor(
                raw_audio=waveform.numpy(),
                sampling_rate=feature_extractor.sampling_rate,
                return_tensors="pt",
            )
            input_values = mimi_inputs["input_values"].to(device)
            padding_mask = mimi_inputs.get("padding_mask")
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)

            with torch.inference_mode():
                encoded = mimi.encode(
                    input_values,
                    padding_mask=padding_mask,
                    num_quantizers=args.num_quantizers,
                )
            audio_codes = encoded.audio_codes[0].long().cpu()
            if audio_codes.shape[0] != args.num_quantizers:
                raise RuntimeError(
                    f"Mimi returned {audio_codes.shape[0]} codebooks, "
                    f"expected {args.num_quantizers}"
                )

            output_path = output_dir / f"{sample_id}.pt"
            torch.save(
                {
                    "id": sample_id,
                    "text": text,
                    "audio": audio_path,
                    "text_hidden": text_hidden.to(torch.float16),
                    "text_token_ids": text_token_ids,
                    "audio_codes": audio_codes,
                    "sampling_rate": feature_extractor.sampling_rate,
                    "num_quantizers": args.num_quantizers,
                },
                output_path,
            )
            destination.write(
                json.dumps(
                    {
                        "id": sample_id,
                        "path": str(output_path.resolve()),
                        "text_tokens": text_hidden.shape[0],
                        "audio_frames": audio_codes.shape[-1],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            destination.flush()
            print(
                f"[{index + 1}] {sample_id}: text={text_hidden.shape[0]} tokens, "
                f"audio={audio_codes.shape[-1]} frames"
            )


if __name__ == "__main__":
    main()
