import argparse
import torch
import tiktoken

from model.gpt import GPT, GPTConfig


def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        new_state_dict[k] = v
    return new_state_dict


def apply_repetition_penalty(logits, prev_tokens, penalty):
    if penalty is None or penalty <= 1.0:
        return logits

    used_tokens = set(prev_tokens)
    for token_id in used_tokens:
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
    return logits


def get_banned_tokens(prev_tokens, ngram_size):
    if ngram_size <= 0:
        return []
    if len(prev_tokens) + 1 < ngram_size:
        return []

    ngrams = {}
    for i in range(len(prev_tokens) - ngram_size + 1):
        prefix = tuple(prev_tokens[i:i + ngram_size - 1])
        next_token = prev_tokens[i + ngram_size - 1]
        ngrams.setdefault(prefix, set()).add(next_token)

    current_prefix = tuple(prev_tokens[-(ngram_size - 1):])
    return list(ngrams.get(current_prefix, []))


def top_k_top_p_filter(logits, top_k=None, top_p=None):
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, -float("inf")), logits)

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_remove = cumulative_probs > top_p
        sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
        sorted_remove[:, 0] = False

        remove_indices = sorted_indices[0, sorted_remove[0]]
        logits[0, remove_indices] = -float("inf")

    return logits


@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    block_size,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    stop_eot,
    eot_token,
):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        prev_tokens = idx[0].tolist()

        logits = apply_repetition_penalty(logits, prev_tokens, repetition_penalty)

        banned_tokens = get_banned_tokens(prev_tokens, no_repeat_ngram_size)
        if banned_tokens:
            logits[:, banned_tokens] = -float("inf")

        logits = logits / max(temperature, 1e-6)
        logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_id), dim=1)

        if stop_eot and next_id.item() == eot_token:
            break

    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_k", type=int, default=60)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--stop_eot", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])

    model = GPT(config)
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=True)
    model.to(device)
    model.eval()

    print(f"number of parameters: {model.get_num_params() / 1e6:.2f}M")

    enc = tiktoken.get_encoding("gpt2")
    prompt = args.prompt.replace("\\n", "\n")
    input_ids = enc.encode(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=args.max_new_tokens,
        block_size=config.block_size,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        stop_eot=args.stop_eot,
        eot_token=enc.eot_token,
    )

    text = enc.decode(out[0].tolist())
    if args.stop_eot:
        text = text.split("<|endoftext|>")[0]

    print(text)


if __name__ == "__main__":
    main()
