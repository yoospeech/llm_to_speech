from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from tts_pipeline.llmvox_transformer import LLMVoxDecoder


@dataclass
class StreamingDepthTTSConfig:
    text_dim: int = 2048
    acoustic_dim: int = 2048
    lfm_hidden_dim: int = 2048
    depth_dim: int = 768
    num_codebooks: int = 4
    codebook_size: int = 2048
    num_layers: int = 4
    num_heads: int = 8
    max_frames: int = 2048
    dropout: float = 0.0
    bias: bool = False
    freeze_lfm: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class StreamingDepthTTS(nn.Module):
    """LLMVoX temporal conditioning plus autoregressive Mimi depth decoding."""

    def __init__(self, config: StreamingDepthTTSConfig) -> None:
        super().__init__()
        self.config = config
        if not (config.text_dim == config.acoustic_dim == config.lfm_hidden_dim):
            raise ValueError("Without an input adapter, text/acoustic/LFM dimensions must match")
        self.text_pad = nn.Parameter(torch.zeros(config.lfm_hidden_dim))
        if config.lfm_hidden_dim % config.num_codebooks:
            raise ValueError("LFM hidden dimension must be divisible by codebook count")
        self.local_sos = config.codebook_size + 1
        self.local_embedding = nn.Embedding(config.codebook_size + 2, config.depth_dim)
        # Moshi-style shared temporal context: one context vector is added to
        # every depth position in the current acoustic frame.
        self.temporal_to_local = nn.Linear(
            config.lfm_hidden_dim, config.depth_dim,
            bias=config.bias,
        )
        self.depth_decoder = LLMVoxDecoder(
            dim=config.depth_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_length=config.num_codebooks,
            dropout=config.dropout,
            bias=config.bias,
        )
        # Codebook 0 owns one additional EOS symbol. EOS is never sent to Mimi.
        self.code_heads = nn.ModuleList([
                nn.Linear(
                config.depth_dim,
                config.codebook_size + (1 if q == 0 else 0),
                bias=False,
            )
            for q in range(config.num_codebooks)
        ])

    def _text_at_audio_rate(
        self, text_hidden: torch.Tensor, text_mask: torch.Tensor, frames: int
    ) -> torch.Tensor:
        projected = text_hidden
        batch = projected.shape[0]
        aligned = self.text_pad.view(1, 1, -1).expand(batch, frames, -1).clone()
        for index in range(batch):
            text_length = min(int(text_mask[index].sum().item()), frames)
            aligned[index, :text_length] = projected[index, :text_length]
        return aligned

    def _temporal_states(
        self,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        frames = codes.shape[-1]
        text_condition = self._text_at_audio_rate(text_hidden, text_mask, frames)
        return text_condition.to(self.temporal_to_local.weight.dtype)

    def _depth_inputs(self, temporal: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        batch, frames, _ = temporal.shape
        shared_condition = self.temporal_to_local(temporal).unsqueeze(2)
        global_condition = shared_condition.expand(
            batch, frames, self.config.num_codebooks, -1
        )
        local_ids = codes.transpose(1, 2).clone()
        local_ids[:, :, 1:] = codes[:, :-1].transpose(1, 2)
        local_ids[:, :, 0] = self.local_sos
        # Ignored EOS-depth and padded targets are never supervised.
        local_ids = local_ids.clamp_min(0)
        return self.local_embedding(local_ids) + global_condition

    def forward(
        self,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor,
        codes: torch.Tensor,
        frame_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del frame_lengths  # used by the loss to mask padded frames
        temporal = self._temporal_states(text_hidden, text_mask, codes)
        batch, frames, _ = temporal.shape
        depth_inputs = self._depth_inputs(temporal, codes)
        depth_hidden = self.depth_decoder(
            depth_inputs.reshape(batch * frames, self.config.num_codebooks, -1)
        ).view(batch, frames, self.config.num_codebooks, -1)
        logits = [
            self.code_heads[q](depth_hidden[:, :, q])
            for q in range(self.config.num_codebooks)
        ]
        return {
            "code_logits": logits,
        }

    @torch.inference_mode()
    def generate(
        self,
        text_hidden: torch.Tensor,
        text_mask: torch.Tensor,
        max_frames: int = 1000,
        min_frames: int | None = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        if text_hidden.shape[0] != 1:
            raise ValueError("streaming generation currently supports batch_size=1")
        text_length = int(text_mask[0].sum().item())
        min_frames = text_length if min_frames is None else min_frames
        eos_token_id = self.config.codebook_size
        generated = torch.empty(
            1,
            self.config.num_codebooks,
            0,
            dtype=torch.long,
            device=text_hidden.device,
        )

        for _ in range(max_frames):
            placeholder = torch.cat(
                [
                    generated,
                    torch.zeros(
                        1,
                        self.config.num_codebooks,
                        1,
                        dtype=torch.long,
                        device=generated.device,
                    ),
                ],
                dim=-1,
            )
            temporal = self._temporal_states(text_hidden, text_mask, placeholder)[:, -1:]
            frame_codes = []
            shared_condition = self.temporal_to_local(temporal)
            global_condition = shared_condition.expand(
                1, 1, self.config.num_codebooks, -1
            ).reshape(1, self.config.num_codebooks, -1)
            local_ids = torch.full((1, 1), self.local_sos, device=generated.device, dtype=torch.long)
            for q in range(self.config.num_codebooks):
                depth = self.local_embedding(local_ids) + global_condition[:, :q + 1]
                depth_hidden = self.depth_decoder(depth)
                logits = self.code_heads[q](depth_hidden[:, q])
                if q == 0 and generated.shape[-1] < min_frames:
                    logits[:, eos_token_id] = -torch.inf
                if temperature > 0:
                    code = torch.multinomial(torch.softmax(logits / temperature, -1), 1)[:, 0]
                else:
                    code = logits.argmax(-1)
                if q == 0 and code.item() == eos_token_id:
                    return generated
                frame_codes.append(code)
                if q + 1 < self.config.num_codebooks:
                    local_ids = torch.cat([local_ids, code[:, None]], dim=1)

            frame = torch.stack(frame_codes, dim=1).unsqueeze(-1)
            generated = torch.cat([generated, frame], dim=-1)
        return generated
