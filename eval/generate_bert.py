import argparse
import re
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast
from model.gpt import GPT, GPTConfig


def clean_bert_decode(text):
    text = text.replace(" ##", "")
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。！？；：、“”‘’（）《》])", r"\1", text)
    text = re.sub(r"([，。！？；：、“”‘’（）《》])\s+", r"\1", text)
    return text.strip()


def calc_banned_ngram_tokens(tokens, n):
    if n <= 0 or len(tokens) < n - 1:
        return set()

    ngram_dict = {}
    for i in range(len(tokens) - n + 1):
        prefix = tuple(tokens[i:i + n - 1])
        nxt = tokens[i + n - 1]
        ngram_dict.setdefault(prefix, set()).add(nxt)

    current_prefix = tuple(tokens[-(n - 1):])
    return ngram_dict.get(current_prefix, set())


@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    temperature=0.8,
    top_k=40,
    eot_id=None,
    repetition_penalty=1.15,
    no_repeat_ngram_size=4,
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # repetition penalty
        if repetition_penalty is not None and repetition_penalty > 1.0:
            used_tokens = set(idx[0].tolist())
            for token_id in used_tokens:
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

        # no repeat ngram
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            tokens = idx[0].tolist()
            banned = calc_banned_ngram_tokens(tokens, no_repeat_ngram_size)
            for token_id in banned:
                logits[0, token_id] = -float("inf")

        logits = logits / temperature

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)

        if eot_id is not None and next_id.item() == eot_id:
            break

    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer_name", default="bert-base-chinese")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only_completion", action="store_true")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    checkpoint = torch.load(args.ckpt, map_location=device)

    model_args = checkpoint.get("model_args", {
        "vocab_size": 21128,
        "block_size": 1024,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "dropout": 0.0,
        "bias": True,
    })

    config = GPTConfig(**model_args)
    model = GPT(config)

    state_dict = checkpoint["model"]
    for prefix in ["_orig_mod.", "module."]:
        if any(k.startswith(prefix) for k in state_dict.keys()):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizerFast.from_pretrained(
        args.tokenizer_name,
        model_max_length=1000000000,
    )

    eot_id = tokenizer.sep_token_id
    input_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    idx = torch.tensor(input_ids, dtype=torch.long, device=device)[None, :]

    out = generate(
        model,
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eot_id=eot_id,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    out_ids = out[0].tolist()

    if args.only_completion:
        out_ids = out_ids[len(input_ids):]

    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    print(clean_bert_decode(text))


if __name__ == "__main__":
    main()
