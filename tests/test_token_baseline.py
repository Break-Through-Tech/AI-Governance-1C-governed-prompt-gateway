"""Validate text preservation, alignment and cost arithmetic for issue #8."""
import csv
from decimal import Decimal
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from measure_token_baseline import count_tokens, measure


class TokenBaselineTests(unittest.TestCase):
    def test_literal_unicode_multiline_duplicates_and_costs(self):
        prompts = ['Hello, world!', '你好\n<|endoftext|>', 'Hello, world!']
        with tempfile.TemporaryDirectory() as d:
            source, output = Path(d) / 'test.csv', Path(d) / 'out'
            with source.open('w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['user_input', 'label'])
                writer.writerows((p, 'safe') for p in prompts)
            before = source.read_bytes()
            report = measure(source, output)
            with (output / 'token_baseline.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(report['input_sha256'], hashlib.sha256(before).hexdigest())
            self.assertEqual([r['test_row'] for r in rows], ['0', '1', '2'])
            self.assertEqual(len({r['prompt_id'] for r in rows}), 3)
            self.assertEqual(rows[0]['prompt_sha256'], rows[2]['prompt_sha256'])
            self.assertEqual([r['prompt_sha256'] for r in rows],
                             [hashlib.sha256(p.encode()).hexdigest() for p in prompts])
            self.assertEqual(count_tokens('Hello, world!'), 4)
            self.assertEqual(report['prompt_count'], 3)
            self.assertEqual(sum(Decimal(r['baseline_input_cost_usd']) for r in rows),
                             Decimal(report['total_baseline_input_cost_usd']))
            self.assertEqual(Decimal(rows[0]['baseline_input_cost_usd']), Decimal('0.0000006'))

    def test_rejects_bad_inputs_before_outputs(self):
        for text in ['user_input,label\n', 'prompt,label\nHi,safe\n',
                     'user_input,label\n,safe\n', 'user_input,label\nHi,unknown\n',
                     'user_input,label\nHi\n', 'user_input,label\nHi,safe,extra\n']:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as d:
                source, output = Path(d) / 'test.csv', Path(d) / 'out'
                source.write_text(text)
                with self.assertRaises(ValueError):
                    measure(source, output)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
