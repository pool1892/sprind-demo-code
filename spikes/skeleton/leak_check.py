"""Search everything the agent can reach, or has produced, for words that name the truth.

    python leak_check.py MAIN_RUN_DIR WORKSPACE             before the first agent call
    python leak_check.py MAIN_RUN_DIR WORKSPACE --after     also the saved agent runs

Before: the workspace (see export.leak_check), the prompt, the answer schema and
the tool server's source, which is readable inside the sandbox. The words are the
fault kinds of the run, a list of telling words, and the canary token that exists
only in the truth directory.

After: the same words in what each agent run left behind (its command line, the
event stream with every tool result, the tool call log). There only the fault
kinds and the canary count as a leak: the agent's own prose may well say "delay".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import diagnose
import export

main_run, ws = Path(sys.argv[1]), Path(sys.argv[2])
words = export.truth_words(main_run)
texts = {
    "system prompt": diagnose.SYSTEM,
    "prompt": diagnose.PROMPT,
    "answer schema": json.dumps(diagnose.ANSWER_SCHEMA),
    "tool server source": (diagnose.HERE / "fleet_tools.py").read_text(),
}
result = export.leak_check(ws, words, texts)
print(f"searched for {len(words)} words: {', '.join(w if not w.startswith('CANARY') else 'the canary token' for w in words)}")
print(f"in {result['searched']['files']} files ({result['searched']['bytes'] / 1e6:.1f} MB), "
      f"{result['searched']['names']} names, {result['searched']['parquet_text_values']} distinct text values, "
      f"{result['searched']['video_tags']} video tags, and {len(texts)} texts given to the agent")
print(f"hits before the agent runs: {len(result['hits'])}")
for h in result["hits"]:
    print("  ", h)
out = {"before": result}

if "--after" in sys.argv:
    strict = [w for w in words if w.startswith("CANARY")]
    for line in (main_run / "truth" / "episodes.jsonl").read_text().splitlines():
        for f in json.loads(line).get("faults") or []:
            if f["kind"] not in strict:
                strict.append(f["kind"])
    hits, files = [], 0
    for answer in sorted((diagnose.BASE / "agent").glob("*.json")):
        run_dir = Path(json.loads(answer.read_text())["run_dir"])
        for p in sorted(run_dir.rglob("*")):
            if p.is_file() and not p.is_symlink():
                files += 1
                data = p.read_bytes()
                for w in strict:
                    if w.encode() in data:
                        hits.append({"where": str(p), "word": w if not w.startswith("CANARY") else "canary"})
    print(f"after the agent runs: {files} files of the saved runs searched for the canary and "
          f"{len(strict) - 1} fault kinds, hits: {len(hits)}")
    for h in hits:
        print("  ", h)
    out["after"] = {"files": files, "hits": hits}

(diagnose.BASE / "leak_check.json").write_text(json.dumps(out, indent=1))
sys.exit(1 if out["before"]["hits"] or out.get("after", {}).get("hits") else 0)
