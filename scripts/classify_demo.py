"""Interactive demo: type a prompt, see the v2 classifier's label/risk_score
and the resulting governance pass/block decision, live (issue #18)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classify_prompt import classify
from scripts.governance import apply_governance


def run_demo(log_path):
    print("Type a prompt and press Enter to classify it. Empty line, 'quit', or Ctrl+D to stop.\n")
    prompt_number = 0
    while True:
        try:
            prompt = input('> ').strip()
        except EOFError:
            print()
            break
        if not prompt or prompt.lower() in ('quit', 'exit'):
            break

        prompt_number += 1
        output = classify(prompt)
        decision = apply_governance(output, prompt_id=f'demo-{prompt_number}', log_path=log_path)
        print(f"  label={decision['label']:<10} risk_score={decision['risk_score']:.3f}  "
              f"-> {decision['action'].upper()}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-path', type=Path,
                        default=ROOT / 'data/processed/governance/classifier_v2_demo.jsonl')
    args = parser.parse_args()
    run_demo(args.log_path)
    print(f'Decisions logged to {args.log_path}')


if __name__ == '__main__':
    main()
