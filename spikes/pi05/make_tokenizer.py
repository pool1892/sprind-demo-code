"""Rebuild the PaliGemma tokenizer from its public file and point the local checkpoints at it.

    ~/.cache/robots-probes/pi05/venv/bin/python spikes/pi05/make_tokenizer.py

pi0 and pi0.5 name `google/paligemma-3b-pt-224` as their tokenizer. That Hugging
Face repository is gated (Google's licence and a login token), and this machine
has neither. The original openpi code reads the same SentencePiece model from a
public file instead:
https://storage.googleapis.com/big_vision/paligemma_tokenizer.model, which
`~/.cache/robots-probes/pi05/tokenizer/src/` holds (downloaded with curl).

Recipe, as in spikes/baseline: put the file into a directory as `tokenizer.model`
with a `tokenizer_config.json` naming `GemmaTokenizer` (start token added, no end
token), load that with transformers' AutoTokenizer and save it. Then check:

1. on 200 prompts in pi0.5's format (instruction, 8 state numbers in 256 bins,
   "Action: ") and 50 in pi0's format (instruction and a newline), that the
   rebuilt tokenizer gives the same ids as the SentencePiece file read directly
   with the start token in front, which is what openpi does;
2. how its files differ from the baseline spike's rebuild, part by part
   (tokenizer/compare_with_baseline.json; `--compare-only` writes just that).

Finally writes `policy_preprocessor.json` into each local checkpoint with the
tokenizer name replaced by this directory; the original stays beside it as
`policy_preprocessor.json.orig`. Writes tokenizer/check.json. No GPU, no simulator.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

OUT = Path.home() / ".cache" / "robots-probes" / "pi05"
TOK = OUT / "tokenizer"
SRC = TOK / "src" / "paligemma_tokenizer.model"
BASELINE_TOK = Path.home() / ".cache" / "robots-probes" / "baseline" / "paligemma-tokenizer-local" / "tokenizer.json"
LIBERO_TASKS = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the alphabet soup and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "put the wine bottle on top of the cabinet",
    "open the top drawer and put the bowl inside",
    "turn on the stove and put the moka pot on it",
]


def compare_with_baseline(local: Path) -> dict:
    """Which parts of the two rebuilds' files differ: tokenizer.json key by key, tokenizer_config.json entry by entry."""
    base_dir = BASELINE_TOK.parent
    out: dict = {"tokenizer_json": {}, "tokenizer_config_json": {}}
    a, b = json.loads((local / "tokenizer.json").read_text()), json.loads(BASELINE_TOK.read_text())
    for k in sorted(set(a) | set(b)):
        out["tokenizer_json"][k] = "same" if a.get(k) == b.get(k) else {"this": a.get(k), "baseline": b.get(k)}
    ca = json.loads((local / "tokenizer_config.json").read_text())
    cb = json.loads((base_dir / "tokenizer_config.json").read_text())
    for k in sorted(set(ca) | set(cb)):
        if ca.get(k) != cb.get(k):
            out["tokenizer_config_json"][k] = {"this": ca.get(k), "baseline": cb.get(k)}
    out["tokenizer_config_json_same_entries"] = sorted(k for k in set(ca) & set(cb) if ca[k] == cb[k])
    return out


def main():
    import sys

    if "--compare-only" in sys.argv:  # compare the existing rebuild with the baseline spike's; build nothing
        rep = compare_with_baseline(TOK / "paligemma-tokenizer-local")
        (TOK / "compare_with_baseline.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps({"tokenizer_json": {k: (v if v == "same" else "differs") for k, v in rep["tokenizer_json"].items()},
                          "tokenizer_config_json_differs": rep["tokenizer_config_json"]}, indent=1))
        return
    import sentencepiece as spm
    from transformers import AutoTokenizer

    stage, local = TOK / "stage", TOK / "paligemma-tokenizer-local"
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC, stage / "tokenizer.model")
    (stage / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "GemmaTokenizer", "add_bos_token": True, "add_eos_token": False,
        "bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>",
    }))
    tok = AutoTokenizer.from_pretrained(str(stage))
    tok.save_pretrained(str(local))
    tok = AutoTokenizer.from_pretrained(str(local))
    sp = spm.SentencePieceProcessor(model_file=str(SRC))

    rng = random.Random(0)
    prompts = []
    for i in range(200):  # pi0.5: "Task: ..., State: b1 ... b8;\nAction: "
        text = rng.choice(LIBERO_TASKS)
        state = " ".join(str(rng.randrange(256)) for _ in range(8))
        prompts.append(f"Task: {text}, State: {state};\nAction: ")
    for i in range(50):  # pi0: the instruction and a newline
        prompts.append(rng.choice(LIBERO_TASKS) + "\n")
    mismatches = []
    for p in prompts:
        a = tok(p)["input_ids"]
        b = [sp.bos_id()] + sp.encode(p)
        if a != b:
            mismatches.append({"prompt": p, "hf": a, "sp": b})
    padded = tok(prompts[0], padding="max_length", max_length=200, padding_side="right")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    report = {
        "source": str(SRC), "source_sha256": sha(SRC), "prompts": len(prompts), "mismatches": len(mismatches),
        "first_mismatches": mismatches[:3], "pad_id": tok.pad_token_id, "bos_id": tok.bos_token_id,
        "padded_len": len(padded["input_ids"]), "padded_tail": padded["input_ids"][-3:],
        "tokenizer_json_sha256": sha(local / "tokenizer.json"),
        "baseline_tokenizer_json_sha256": sha(BASELINE_TOK) if BASELINE_TOK.exists() else None,
    }
    report["same_as_baseline_rebuild"] = report["tokenizer_json_sha256"] == report["baseline_tokenizer_json_sha256"]
    if BASELINE_TOK.exists():
        (TOK / "compare_with_baseline.json").write_text(json.dumps(compare_with_baseline(local), indent=1))

    for name in ("pi05_libero_finetuned_v044", "pi0_libero_finetuned_v044"):
        d = OUT / "ckpt" / name
        pre = d / "policy_preprocessor.json"
        if not pre.exists() and not (d / "policy_preprocessor.json.orig").exists():
            report[name] = "not downloaded"
            continue
        orig = d / "policy_preprocessor.json.orig"
        if not orig.exists():
            shutil.copyfile(pre, orig)
        cfg = json.loads(orig.read_text())
        n = 0
        for step in cfg["steps"]:
            if step["registry_name"] == "tokenizer_processor":
                assert step["config"]["tokenizer_name"] == "google/paligemma-3b-pt-224", step
                step["config"]["tokenizer_name"] = str(local)
                n += 1
        assert n == 1, name
        pre.write_text(json.dumps(cfg, indent=2))
        report[name] = f"tokenizer_name -> {local}"
    (TOK / "check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "first_mismatches"}, indent=1))


if __name__ == "__main__":
    main()
