import json
import math
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, MimiModel

class RawTTSDataset(Dataset):
    """Read text/audio pairs without precomputing model features."""

    def __init__(self, manifest: str, max_duration_seconds: float | None = None) -> None:
        manifest_path = Path(manifest).resolve()
        self.base_dir = manifest_path.parent
        with manifest_path.open(encoding="utf-8") as handle:
            if manifest_path.suffix.lower() == '.json':
                self.items = json.load(handle)
            else:
                self.items = [json.loads(line) for line in handle if line.strip()]
        if max_duration_seconds is not None:
            self.items = [
                item for item in self.items
                if float(item["duration_seconds"]) <= max_duration_seconds
            ]
        if not self.items:
            raise ValueError("No samples remain after applying the duration limit")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, str]:
        item = self.items[index]
        audio_path = Path(item["audio"])
        if not audio_path.is_absolute():
            audio_path = self.base_dir / audio_path
        return {
            "id": str(item.get("id", index)),
            "text": item["text"],
            "audio": str(audio_path.resolve()),
        }


class OnlineCodecCollator:
    """Keep text raw and extract only frozen Mimi codes for each batch."""

    def __init__(
        self,
        mimi_model: str = "kyutai/mimi",
        num_quantizers: int = 4,
        device: str = "cuda",
        max_frames: int = 2048,
    ) -> None:
        if max_frames < 2:
            raise ValueError("max_frames must leave room for at least one audio frame and EOS")
        self.device = torch.device(device)
        self.num_quantizers = num_quantizers
        self.max_frames = max_frames
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(mimi_model)
        self.mimi = MimiModel.from_pretrained(mimi_model).to(self.device).eval()
        self.eos_token_id = int(self.mimi.config.codebook_size)
        self.sampling_rate = int(self.feature_extractor.sampling_rate)
        self.samples_per_frame = self.sampling_rate / 12.5

        self.mimi.requires_grad_(False)

    def _load_audio(self, path: str) -> torch.Tensor:
        audio, source_rate = sf.read(path, dtype='float32', always_2d=True)
        waveform = torch.from_numpy(audio).mean(dim=1)
        if source_rate != self.sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform, source_rate, self.sampling_rate
            )
        # One temporal slot is reserved for EOS, so only encode the audio
        # duration that can contribute supervised Mimi frames.
        max_audio_samples = math.floor((self.max_frames - 1) * self.samples_per_frame)
        return waveform[:max_audio_samples].clamp(-1.0, 1.0)

    @torch.inference_mode()
    def __call__(self, samples: list[dict[str, str]]) -> dict:
        waveforms = [self._load_audio(item["audio"]) for item in samples]
        mimi_inputs = self.feature_extractor(
            raw_audio=[waveform.numpy() for waveform in waveforms],
            sampling_rate=self.sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        input_values = mimi_inputs["input_values"].to(self.device)
        padding_mask = mimi_inputs.get("padding_mask")
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)
        encoded = self.mimi.encode(
            input_values,
            padding_mask=padding_mask,
            num_quantizers=self.num_quantizers,
        )
        all_codes = encoded.audio_codes.long().cpu()
        if all_codes.ndim != 3 or all_codes.shape[1] != self.num_quantizers:
            raise RuntimeError(
                f"Unexpected Mimi code shape {tuple(all_codes.shape)}; expected [B, "
                f"{self.num_quantizers}, T]"
            )

        batch = len(samples)
        frame_lengths = torch.tensor(
            [
                min(
                    math.ceil(waveform.numel() / self.samples_per_frame),
                    all_codes.shape[-1],
                    self.max_frames - 1,
                )
                for waveform in waveforms
            ],
            dtype=torch.long,
        )
        # Use the longest item in this batch, capped by max_frames. One position
        # after each item's last audio frame is reserved for EOS on codebook 0.
        batch_frames = min(int(frame_lengths.max().item()) + 1, self.max_frames)
        codes = torch.full(
            (batch, self.num_quantizers, batch_frames), -100, dtype=torch.long
        )

        for index in range(batch):
            frame_length = int(frame_lengths[index].item())
            codes[index, :, :frame_length] = all_codes[index, :, :frame_length]
            codes[index, 0, frame_length] = self.eos_token_id

        return {
            "texts": [item["text"] for item in samples],
            "codes": codes,
            "frame_lengths": frame_lengths,
        }
