"""Training entry point: python -m scripts.train_depth_tts."""
import argparse
import json
import math
import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tts_pipeline.losses import streaming_depth_tts_loss
from tts_pipeline.online_data import OnlineCodecCollator, RawTTSDataset
from tts_pipeline.streaming_depth_tts import StreamingDepthTTS, StreamingDepthTTSConfig
from tts_pipeline.text_embedder import LFMTextEmbedder
from tts_pipeline.libritts_data import LibriTTSRawDataset


class TTSDataset(Dataset):
    def __init__(self, manifest: str) -> None:
        with open(manifest, encoding="utf-8") as handle:
            self.items = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return torch.load(self.items[index]["path"], map_location="cpu", weights_only=False)


def collate(samples: list[dict], max_frames: int = 2048) -> dict[str, torch.Tensor]:
    if max_frames < 2:
        raise ValueError("max_frames must leave room for at least one audio frame and EOS")
    batch = len(samples)
    hidden_dim = samples[0]["text_hidden"].shape[-1]
    codebooks = samples[0]["audio_codes"].shape[0]
    max_text = max(item["text_hidden"].shape[0] for item in samples)
    batch_frames = min(
        max(item["audio_codes"].shape[-1] for item in samples) + 1,
        max_frames,
    )
    text_hidden = torch.zeros(batch, max_text, hidden_dim, dtype=torch.float32)
    text_mask = torch.zeros(batch, max_text, dtype=torch.bool)
    codes = torch.full((batch, codebooks, batch_frames), -100, dtype=torch.long)
    frame_lengths = torch.empty(batch, dtype=torch.long)
    for i, item in enumerate(samples):
        text_len = item["text_hidden"].shape[0]
        frame_len = min(item["audio_codes"].shape[-1], max_frames - 1)
        text_hidden[i, :text_len] = item["text_hidden"].float()
        text_mask[i, :text_len] = True
        codes[i, :, :frame_len] = item["audio_codes"][:, :frame_len]
        codes[i, 0, frame_len] = 2048
        frame_lengths[i] = frame_len
    return {
        "text_hidden": text_hidden,
        "text_mask": text_mask,
        "codes": codes,
        "frame_lengths": frame_lengths,
    }


def embed_raw_texts(
    texts: list[str], embedder: LFMTextEmbedder
) -> tuple[torch.Tensor, torch.Tensor]:
    features = [embedder(text)[0] for text in texts]
    max_text = max(hidden.shape[0] for hidden in features)
    hidden_dim = features[0].shape[-1]
    text_hidden = torch.zeros(len(features), max_text, hidden_dim, dtype=torch.float32)
    text_mask = torch.zeros(len(features), max_text, dtype=torch.bool)
    for index, hidden in enumerate(features):
        length = hidden.shape[0]
        text_hidden[index, :length] = hidden
        text_mask[index, :length] = True
    return text_hidden, text_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--libritts-root")
    parser.add_argument("--splits", nargs="+", default=["train-clean-100", "train-clean-360"])
    parser.add_argument("--text-type", choices=["normalized", "original"], default="normalized")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--tensorboard-logdir", default=None)
    parser.add_argument("--save-top-k", type=int, default=3, help="Number of best validation checkpoints to retain")
    parser.add_argument("--valid-ratio", type=float, default=0.01, help="Utterance-level validation fraction; 0 disables validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-samples", type=int, default=0, help="Limit dataset to first N samples for a smoke run")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=2048)
    parser.add_argument(
        "--max-audio-seconds", type=float, default=None,
        help="Limit audio duration; converted at Mimi's 12.5 frames/second plus one EOS slot",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument(
        "--online",
        action="store_true",
        help="Keep text raw and extract Mimi codes during training",
    )
    parser.add_argument("--lfm-model", default="./LFM3-1.7B")
    parser.add_argument("--mimi-model", default="kyutai/mimi")
    parser.add_argument("--num-quantizers", type=int, default=4)
    parser.add_argument(
        "--extractor-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def split_dataset(dataset: Dataset, valid_ratio: float, seed: int):
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid-ratio must be in [0, 1)")
    if len(dataset) == 0 or (valid_ratio > 0 and len(dataset) < 2):
        raise ValueError("Need at least two samples for train/valid splitting (or use --valid-ratio 0)")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    n_valid = max(1, int(len(dataset) * valid_ratio)) if valid_ratio else 0
    valid_indices, train_indices = indices[:n_valid], indices[n_valid:]
    return Subset(dataset, train_indices), Subset(dataset, valid_indices)


def prepare_batch(batch, text_embedder, device, online):
    if online:
        batch = dict(batch)
        batch["text_hidden"], batch["text_mask"] = embed_raw_texts(batch.pop("texts"), text_embedder)
    return {key: value.to(device, non_blocking=True).clone() for key, value in batch.items()}


@torch.inference_mode()
def validate(model, loader, text_embedder, device, online, num_codebooks, distributed):
    model.eval()
    totals = torch.zeros(num_codebooks, dtype=torch.float64, device=device)
    counts = torch.zeros_like(totals)
    corrects = torch.zeros_like(totals)
    for batch in loader:
        batch = prepare_batch(batch, text_embedder, device, online)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(**batch)
            for q, logits in enumerate(output["code_logits"]):
                targets = batch["codes"][:, q].reshape(-1)
                totals[q] += torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), targets,
                    ignore_index=-100, reduction="sum",
                )
                counts[q] += (targets != -100).sum()
                valid = targets != -100
                corrects[q] += (logits.float().reshape(-1, logits.shape[-1])[valid].argmax(-1) == targets[valid]).sum()
    if distributed:
        dist.all_reduce(totals)
        dist.all_reduce(counts)
        dist.all_reduce(corrects)
    if (counts == 0).any():
        raise ValueError("Validation contains a codebook without valid targets")
    return (totals / counts).mean().item(), (corrects / counts).tolist()


def main() -> None:
    args = parse_args()
    if not 0 <= args.valid_ratio < 1:
        raise ValueError("valid-ratio must be in [0, 1)")
    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be positive")
    if args.save_top_k < 1:
        raise ValueError("save-top-k must be positive")
    if args.max_audio_seconds is not None:
        if args.max_audio_seconds <= 0:
            raise ValueError("max-audio-seconds must be positive")
        args.max_frames = math.ceil(args.max_audio_seconds * 12.5) + 1
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA with this script")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    torch.manual_seed(args.seed)
    if args.libritts_root:
        args.online = True
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_logdir or str(output_dir / "tensorboard"))
    else:
        writer = None
    if distributed:
        dist.barrier()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if args.online:
        dataset = (
            LibriTTSRawDataset(
                args.libritts_root, args.splits, args.text_type, args.max_audio_seconds
            )
            if args.libritts_root else RawTTSDataset(args.manifest, args.max_audio_seconds)
        )
        online_collator = OnlineCodecCollator(
            mimi_model=args.mimi_model,
            num_quantizers=args.num_quantizers,
            device=str(device) if args.extractor_device == "cuda" else args.extractor_device,
            max_frames=args.max_frames,
        )
        text_embedder = LFMTextEmbedder(args.lfm_model, str(device))
        collate_fn = online_collator
        num_workers = 0
        config = StreamingDepthTTSConfig(
            text_dim=2048,
            num_codebooks=args.num_quantizers,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_frames=args.max_frames,
        )
    else:
        dataset = TTSDataset(args.manifest)
        text_embedder = None
        first = dataset[0]
        collate_fn = partial(collate, max_frames=args.max_frames)
        num_workers = args.num_workers
        config = StreamingDepthTTSConfig(
            text_dim=first["text_hidden"].shape[-1],
            num_codebooks=first["audio_codes"].shape[0],
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_frames=args.max_frames,
        )
    if text_embedder is None:
        text_embedder = LFMTextEmbedder(args.lfm_model, str(device))
    config.lfm_hidden_dim = text_embedder.model.config.hidden_size
    if args.smoke_samples > 0:
        dataset = Subset(dataset, range(min(args.smoke_samples, len(dataset))))
    train_dataset, valid_dataset = split_dataset(dataset, args.valid_ratio, args.seed)
    split_info = {
        "seed": args.seed, "valid_ratio": args.valid_ratio,
        "manifest": args.manifest, "libritts_root": args.libritts_root,
        "splits": args.splits, "smoke_samples": args.smoke_samples,
        "max_audio_seconds": args.max_audio_seconds,
        "dataset_size": len(dataset),
        "train_indices": train_dataset.indices, "valid_indices": valid_dataset.indices,
    }
    if rank == 0:
        with (output_dir / "split.json").open("w", encoding="utf-8") as handle:
            json.dump(split_info, handle, indent=2)
        print(
            f"train={len(train_dataset)} valid={len(valid_dataset)} seed={args.seed} "
            f"world_size={dist.get_world_size() if distributed else 1}"
        )
    model = StreamingDepthTTS(config).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )
    train_sampler = DistributedSampler(
        train_dataset, shuffle=True, seed=args.seed, drop_last=False
    ) if distributed else None
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    rank_valid_dataset = (
        Subset(valid_dataset, range(rank, len(valid_dataset), dist.get_world_size()))
        if distributed else valid_dataset
    )
    valid_loader = DataLoader(
        rank_valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    ) if len(valid_dataset) else None
    best_valid_loss = float("inf")
    best_checkpoints: list[tuple[float, Path]] = []
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        running_accuracy = 0.0
        for step, batch in enumerate(loader, 1):
            batch = prepare_batch(batch, text_embedder, device, args.online)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(**batch)
                losses = streaming_depth_tts_loss(
                    code_logits=output["code_logits"],
                    codes=batch["codes"],
                )
                group_start = ((step - 1) // args.gradient_accumulation) * args.gradient_accumulation
                group_size = min(args.gradient_accumulation, len(loader) - group_start)
                scaled_loss = losses["loss"] / group_size
            scaler.scale(scaled_loss).backward()
            if step % args.gradient_accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += losses["loss"].item()
            running_accuracy += losses["accuracy"].item()
            if rank == 0:
                writer.add_scalar("train/loss_step", losses["loss"].item(), (epoch - 1) * len(loader) + step)
                writer.add_scalar("train/accuracy_step", losses["accuracy"].item(), (epoch - 1) * len(loader) + step)
                print(
                    f"epoch={epoch} step={step}/{len(loader)} loss={losses['loss'].item():.4f} "
                    f"codebooks={[round(value, 4) for value in losses['codebook_losses'].tolist()]} "
                    f"acc={[round(value, 4) for value in losses['codebook_accuracies'].tolist()]}"
                )

        validation_model = model.module if distributed else model
        validation_result = validate(
            validation_model, valid_loader, text_embedder, device, args.online,
            config.num_codebooks, distributed,
        ) if valid_loader is not None else None
        valid_loss = validation_result[0] if validation_result is not None else None
        valid_accuracies = validation_result[1] if validation_result is not None else None
        running_tensor = torch.tensor([running, len(loader)], dtype=torch.float64, device=device)
        running_accuracy_tensor = torch.tensor([running_accuracy, len(loader)], dtype=torch.float64, device=device)
        if distributed:
            dist.all_reduce(running_tensor)
            dist.all_reduce(running_accuracy_tensor)
        mean_train_loss = (running_tensor[0] / running_tensor[1]).item()
        mean_train_accuracy = (running_accuracy_tensor[0] / running_accuracy_tensor[1]).item()
        is_best = valid_loss is not None and valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = valid_loss
        checkpoint = {
            "model": validation_model.state_dict(),
            "config": config.to_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "model_type": "llmvox_mimi_depth",
            "valid_loss": valid_loss,
            "best_valid_loss": best_valid_loss,
            "split_file": "split.json",
        }
        if rank == 0:
            torch.save(checkpoint, output_dir / "last.pt")
            if is_best:
                torch.save(checkpoint, output_dir / "best.pt")
            writer.add_scalar("train/loss_epoch", mean_train_loss, epoch)
            writer.add_scalar("train/accuracy_epoch", mean_train_accuracy, epoch)
            if valid_loss is not None:
                writer.add_scalar("valid/loss", valid_loss, epoch)
                writer.add_scalar("valid/accuracy", sum(valid_accuracies) / len(valid_accuracies), epoch)
                for q, accuracy in enumerate(valid_accuracies):
                    writer.add_scalar(f"valid/codebook_{q}_accuracy", accuracy, epoch)
                candidate_path = output_dir / f"best_epoch_{epoch:04d}_valid_{valid_loss:.5f}.pt"
                torch.save(checkpoint, candidate_path)
                best_checkpoints.append((valid_loss, candidate_path))
                best_checkpoints.sort(key=lambda item: item[0])
                stale_checkpoints = best_checkpoints[args.save_top_k:]
                for _, stale_path in stale_checkpoints:
                    stale_path.unlink(missing_ok=True)
                best_checkpoints = best_checkpoints[:args.save_top_k]
            writer.flush()
            print(
                f"epoch={epoch} mean_loss={mean_train_loss:.4f} valid_loss={valid_loss} "
                f"valid_acc={valid_accuracies}"
            )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
