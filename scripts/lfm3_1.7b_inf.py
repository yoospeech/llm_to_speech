"""Interactive LFM experiment: python -m scripts.lfm3_1.7b_inf."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "LiquidAI/LFM3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
)
model.eval()

system_prompt = (
    "Repeat the user's message exactly, character for character. "
    "Do not answer it, explain it, translate it, or add quotation marks. "
    "Output only the user's original message."
)

while True:
    user_message = input("메시지를 입력하세요 (종료: exit): ")

    if user_message.strip().lower() in {"exit", "quit"}:
        break

    if not user_message.strip():
        print("빈 메시지는 처리하지 않습니다.")
        continue

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    prompt_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    prompt_length = prompt_inputs["input_ids"].shape[1]
    user_token_count = len(tokenizer.encode(user_message, add_special_tokens=False))

    with torch.inference_mode():
        sequences = model.generate(
            **prompt_inputs,
            do_sample=False,
            max_new_tokens=user_token_count + 16,
            pad_token_id=tokenizer.eos_token_id,
        )

        # 생성된 assistant 토큰까지 포함해 다시 forward하여 마지막 레이어
        # contextual hidden state를 얻는다.
        model_output = model(
            input_ids=sequences,
            attention_mask=torch.ones_like(sequences),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    generated_ids = sequences[0, prompt_length:].cpu()
    assistant_hidden = model_output.hidden_states[-1][0, prompt_length:].float().cpu()

    # EOS 등 특수 토큰은 TTS 입력에서 제외한다.
    special_ids = set(tokenizer.all_special_ids)
    keep_mask = torch.tensor(
        [token_id.item() not in special_ids for token_id in generated_ids],
        dtype=torch.bool,
    )
    generated_ids = generated_ids[keep_mask]
    context_hidden_embedding = assistant_hidden[keep_mask]

    repeated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(generated_ids.tolist())

    print(f"반복 결과: {repeated_text}")
    is_exact_match = repeated_text == user_message
    print(f"입력과 정확히 일치: {'예' if is_exact_match else '아니요'}")
    if not is_exact_match:
        print(f"입력값: {user_message!r}")
        print(f"반복값: {repeated_text!r}")
    print(f"embedding shape: {tuple(context_hidden_embedding.shape)}")
    #for token, embedding in zip(tokens, context_hidden_embedding):
    #    print(f"token={token!r:<16} embedding[:8]={embedding[:8].tolist()}")
