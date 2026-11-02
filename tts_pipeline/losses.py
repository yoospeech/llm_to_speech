import torch
import torch.nn.functional as F


def streaming_depth_tts_loss(
    code_logits: list[torch.Tensor],
    codes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    per_codebook = []
    accuracies = []
    for q, logits in enumerate(code_logits):
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = codes[:, q].reshape(-1)
        per_codebook.append(
            F.cross_entropy(
                flat_logits, flat_targets,
                ignore_index=-100,
            )
        )
        valid = flat_targets != -100
        accuracies.append(
            (flat_logits[valid].argmax(dim=-1) == flat_targets[valid]).float().mean()
            if valid.any() else flat_logits.new_zeros(())
        )
    codebook_losses = torch.stack(per_codebook)
    return {
        "loss": codebook_losses.mean(),
        "code_loss": codebook_losses.mean(),
        "codebook_losses": codebook_losses,
        "codebook_accuracies": torch.stack(accuracies),
        "accuracy": torch.stack(accuracies).mean(),
    }
