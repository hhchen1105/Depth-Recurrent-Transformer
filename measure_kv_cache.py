"""
Measured key-value-cache footprint of a real depth-24 token backbone
(paper Fig. fig:inference-cost, dashed curves).

Earlier revisions of the figure used an analytical KV-cache term evaluated at
the *reasoning core's* width (d=256), which is not the width of any real
depth-24 model. Here we load GPT-2 medium (N=24 layers, d=1024, 16 heads x 64
head dim) and read the true size of the `past_key_values` tensors.

We charge the reasoning budget only for the thinking tokens, never for the
prompt: the recurrent core carries the same prompt as the backbone does, so
billing the prompt to the reasoning budget would flatter our side. The
thinking-token cost is measured by difference -- one forward pass over the
L-token prompt, one over L+T tokens -- rather than derived from a formula. A
single forward with use_cache=True leaves exactly one cache entry per input
token, so the difference is exactly the T thinking tokens' cache.

(Measuring with generate(max_new_tokens=T) instead would leave L+T-1 entries:
the T-th token is emitted but never fed back. Horizontal CoT must feed all T
thinking tokens back to produce the answer, so T -- not T-1 -- is the right
charge, which is what the forward-pass difference gives.)

The cache is exactly linear in the batch size -- every sequence carries its own
copy -- so we measure at small B, assert linearity holds to the byte, and scale
to the serving batch sizes. This keeps the measurement runnable without a large
GPU while remaining exact rather than estimated.

Writes kv_cache_measured.json next to this file.
"""

import json
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = "gpt2-medium"          # N=24, d=1024 -- the depth-24 backbone in the paper
L = 128                        # prompt length, same as the core's sequence length
T_LIST = [1, 2, 4, 8, 16, 32, 64]
B_LIST = [1, 32, 128, 512]     # B=1 measured; the rest scaled (exact, see docstring)
B_MEASURED = [1, 2, 4]         # measured directly to verify exact linearity in B
DTYPE = torch.float16          # fp16 KV cache, as assumed in the complexity analysis
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_cache_measured.json")


def cache_bytes(past):
    """Total bytes held by the key/value tensors of a generation cache."""
    total, seen, tensors = 0, set(), []
    if hasattr(past, "layers"):                      # transformers >= 5 Cache object
        for layer in past.layers:
            tensors += [layer.keys, layer.values]
    elif hasattr(past, "key_cache"):                 # transformers 4.x Cache object
        tensors = list(past.key_cache) + list(past.value_cache)
    else:                                            # legacy tuple-of-tuples
        for layer in past:
            tensors += list(layer)
    for t in tensors:
        if t is None or id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


def cache_for(model, B, S, vocab):
    """Bytes of KV cache left by one forward pass over B sequences of S tokens."""
    ids = torch.randint(0, vocab, (B, S))
    with torch.no_grad():
        out = model(ids, use_cache=True)
    return cache_bytes(out.past_key_values)


def main():
    cfg = AutoConfig.from_pretrained(MODEL)
    N, d = cfg.n_layer, cfg.n_embd
    print(f"{MODEL}: N={N} layers, d={d}, heads={cfg.n_head}, dtype=fp16")

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).eval()

    prompt_only = {B: cache_for(model, B, L, cfg.vocab_size) for B in B_MEASURED}
    think = {}                       # think[B][T] = bytes attributable to T thinking tokens
    for B in B_MEASURED:
        think[B] = {}
        for T in T_LIST:
            total = cache_for(model, B, L + T, cfg.vocab_size)
            think[B][T] = total - prompt_only[B]
            print(f"  B={B:>3} T={T:>3}  prompt={prompt_only[B]/1e6:7.1f} MB  "
                  f"+thinking={think[B][T]/1e6:7.1f} MB")

    # The cache must be exactly proportional to B; verify before scaling.
    for T in T_LIST:
        for B in B_MEASURED[1:]:
            assert think[B][T] == think[1][T] * B, f"not linear in B at T={T}"
    print("linearity in B verified exactly for every T")

    # And exactly proportional to the number of thinking tokens.
    for T in T_LIST:
        expect = 2 * 1 * T * N * d * 2          # 2 tensors (K,V) x T tokens x N x d x fp16
        assert think[1][T] == expect, f"T={T}: {think[1][T]} != {expect}"
    print("measured thinking-token cache matches 2*B*T*N*d*bytes exactly")

    out = {
        "model": MODEL, "n_layer": N, "d_model": d, "n_head": cfg.n_head,
        "prompt_len": L, "dtype": "fp16", "method": "forward-pass difference",
        "T_list": T_LIST, "B_list": B_LIST, "measured_B": B_MEASURED,
        "prompt_cache_mb": {str(B): round(prompt_only[1] * B / 1e6, 1) for B in B_LIST},
        "thinking_kv_cache_mb": {
            str(B): [round(think[1][T] * B / 1e6, 1) for T in T_LIST] for B in B_LIST
        },
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT}")
    for B in B_LIST:
        print(f"  B={B:>3}: {out['thinking_kv_cache_mb'][str(B)]}")


if __name__ == "__main__":
    main()
