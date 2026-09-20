"""Behavioral checks for source mapping and leakage controls."""
import unittest

from scripts.build_risk_dataset import classify_toxic_chat, finalize_records, make_record


class TaxonomyTests(unittest.TestCase):
    def test_overlap_keeps_jailbreak_precedence(self):
        self.assertEqual(classify_toxic_chat('1', '1'), 'Jailbreak')
        self.assertEqual(classify_toxic_chat('0', '1'), 'Jailbreak')

    def test_source_negatives_and_toxicity(self):
        self.assertEqual(classify_toxic_chat('0', '0'), 'Safe')
        self.assertEqual(classify_toxic_chat('1', '0'), 'Toxic')

    def test_missing_signals_do_not_become_safe(self):
        for toxic, jailbreak in [('', '0'), ('0', ''), ('', ''), ('1', '')]:
            self.assertEqual(classify_toxic_chat(toxic, jailbreak), 'Unknown')
        self.assertEqual(classify_toxic_chat('', '1'), 'Jailbreak')

    def test_invalid_signal_fails(self):
        with self.assertRaises(ValueError):
            classify_toxic_chat('2', '0')

    def test_held_out_overlap_is_not_training_eligible(self):
        rows = [self.row('1', ' A benign request ', 'train'),
                self.row('2', 'a BENIGN request', 'test')]
        finalize_records(rows)
        self.assertTrue(all(r['cross_partition_duplicate'] == 'true' for r in rows))
        self.assertEqual(rows[0]['training_eligible'], 'false')
        self.assertEqual(rows[1]['source_split'], 'test')

    def test_conflicting_labels_require_review(self):
        rows = [self.row('1', 'same text', 'train'),
                self.row('2', 'same text', 'train', 'Toxic')]
        finalize_records(rows)
        self.assertTrue(all(r['label_conflict'] == 'true' for r in rows))
        self.assertTrue(all(r['governance_action'] == 'review' for r in rows))
        self.assertTrue(all(r['training_eligible'] == 'false' for r in rows))
        self.assertEqual([r['label'] for r in rows], ['Safe', 'Toxic'])

    def test_missing_prompt_and_unknown_are_excluded(self):
        rows = [self.row('1', '   ', 'train'), self.row('2', 'text', 'train', 'Unknown')]
        finalize_records(rows)
        self.assertTrue(all(r['training_eligible'] == 'false' for r in rows))
        self.assertTrue(all(r['governance_action'] == 'review' for r in rows))

    def test_valid_train_and_benchmark_roles(self):
        train = self.row('1', 'ordinary request', 'train')
        benchmark = make_record(dataset='jailbreakbench', config='behaviors', split='harmful',
                                member='x.csv', row_number=1, source_id='0',
                                prompt='benchmark request', label='Harmful')
        finalize_records([train, benchmark])
        self.assertEqual(train['training_eligible'], 'true')
        self.assertEqual(benchmark['training_eligible'], 'false')
        self.assertEqual(benchmark['governance_action'], 'block')

    @staticmethod
    def row(identifier, prompt, split, label='Safe'):
        row = make_record(dataset='toxic_chat', config='annotations', split=split, member='x.csv',
                          row_number=int(identifier), source_id=identifier, prompt=prompt, label=label)
        row.update(toxicity='1' if label in ('Toxic', 'Jailbreak') else '0',
                   jailbreaking='1' if label == 'Jailbreak' else '0', human_annotation='True')
        return row


if __name__ == '__main__':
    unittest.main()
