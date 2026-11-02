"""Lazy LibriTTS_R text/audio metadata dataset; waveforms are loaded by the codec collator."""

from pathlib import Path
import wave


class LibriTTSRawDataset:
    def __init__(
        self, root, splits=("train-clean-100", "train-clean-360"),
        text_type="normalized", max_duration_seconds=None,
    ):
        if text_type not in {"normalized", "original"}:
            raise ValueError("text_type must be normalized or original")
        self.text_type = text_type
        self.paths = []
        for split in splits:
            directory = Path(root).resolve() / split
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            self.paths.extend(sorted(directory.glob("*/*/*.wav")))
        if max_duration_seconds is not None:
            kept_paths = []
            for path in self.paths:
                with wave.open(str(path), "rb") as audio:
                    duration = audio.getnframes() / audio.getframerate()
                if duration <= max_duration_seconds:
                    kept_paths.append(path)
            self.paths = kept_paths
        if not self.paths:
            raise ValueError(f"No utterance WAV files found under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        text_path = path.with_suffix(f".{self.text_type}.txt")
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty transcript: {text_path}")
        with wave.open(str(path), "rb") as audio:
            samples = audio.getnframes()
            rate = audio.getframerate()
        if samples <= 0:
            raise ValueError(f"Empty audio: {path}")
        return {
            "id": path.stem,
            "speaker_id": path.parent.parent.name,
            "text": text,
            "audio": str(path),
            "num_samples": samples,
            "sampling_rate": rate,
            "duration_seconds": samples / rate,
        }
