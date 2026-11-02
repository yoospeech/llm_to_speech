import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ECHO_SYSTEM_PROMPT = (
    "Repeat the user's message exactly, character for character. "
    "Do not answer it, explain it, translate it, or add quotation marks. "
    "Output only the user's original message."
)


class LFMTextEmbedder:
    """Extract teacher-forced assistant hidden states without LM sampling."""

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_messages = [
            {"role": "system", "content": ECHO_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prefix = self.tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        full_messages = prefix_messages + [{"role": "assistant", "content": text}]
        full = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        output = self.model(
            **full,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        start = prefix["input_ids"].shape[1]
        token_ids = full["input_ids"][0, start:]
        hidden = output.hidden_states[-1][0, start:]

        special_ids = set(self.tokenizer.all_special_ids)
        keep = torch.tensor(
            [token_id.item() not in special_ids for token_id in token_ids],
            device=token_ids.device,
            dtype=torch.bool,
        )
        return hidden[keep].float().cpu(), token_ids[keep].cpu()
