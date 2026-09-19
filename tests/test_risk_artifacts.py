"""Validate the actual source-to-artifact contract without printing source text."""
import csv
import io
from pathlib import Path
import unittest
import zipfile

from scripts.build_risk_dataset import (
    FIELDS, JUDGE_FIELDS, build_artifacts, finalize_records, load_records, read_member,
)

DATA = Path(__file__).resolve().parents[1] / 'data'


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.judges, cls.manifest = load_records(DATA)
        finalize_records(cls.rows)

    def test_all_source_rows_have_unique_provenance(self):
        self.assertEqual(len(self.rows), 10665)
        self.assertEqual(len({r['record_id'] for r in self.rows}), len(self.rows))
        self.assertEqual(sum(m['rows'] for archive in self.manifest.values()
                             for m in archive['members'].values()), len(self.rows))

    def test_original_prompts_annotations_and_splits_are_preserved(self):
        by_id = {r['record_id']: r for r in self.rows}
        with zipfile.ZipFile(DATA / 'lmsys_toxicchat.zip') as archive:
            for split in ('train', 'test'):
                with archive.open(f'lmsys_toxicchat/{split}.csv') as stream:
                    source_rows = csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig', newline=''))
                    for number, source in enumerate(source_rows, 1):
                        row = by_id[f"toxic_chat:annotations:{split}:{source['conv_id']}"]
                        self.assertTrue(row['prompt'] == source['user_input'])
                        self.assertEqual(row['source_split'], split)
                        self.assertEqual(row['source_row'], str(number))
                        for key in ('toxicity', 'jailbreaking', 'human_annotation'):
                            self.assertEqual(row[key], source[key])
                        expected = ('Jailbreak' if source['jailbreaking'] == '1'
                                    else 'Toxic' if source['toxicity'] == '1' else 'Safe')
                        self.assertEqual(row['label'], expected)

    def test_behavior_goals_not_targets_and_harmful_not_jailbreak(self):
        by_id = {r['record_id']: r for r in self.rows}
        with zipfile.ZipFile(DATA / 'jailbreak.zip') as archive:
            for split in ('benign', 'harmful'):
                with archive.open(f'jailbreak/behaviors_{split}.csv') as stream:
                    for source in csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig', newline='')):
                        row = by_id[f"jailbreakbench:behaviors:{split}:{source['Index']}"]
                        self.assertTrue(row['prompt'] == source['Goal'])
                        self.assertEqual(row['label'], 'Safe' if split == 'benign' else 'Harmful')
                        self.assertEqual(row['training_eligible'], 'false')

    def test_judge_records_are_lossless_and_excluded(self):
        by_id = {r['record_id']: r for r in self.rows}
        with zipfile.ZipFile(DATA / 'jailbreak.zip') as archive:
            sources = read_member(archive, 'jailbreak/judge_comparison_test.csv', JUDGE_FIELDS)
        self.assertEqual(len(self.judges), 300)
        for source, judge in zip(sources, self.judges):
            self.assertTrue({k: judge[k] for k in JUDGE_FIELDS} == source)
            row = by_id[judge['record_id']]
            self.assertTrue(row['prompt'] == source['prompt'])
            self.assertEqual(row['label'], 'Unknown')
            self.assertEqual(row['training_eligible'], 'false')
            self.assertEqual(row['governance_action'], 'review')
        forbidden = {'model_output', 'Target', 'target_response', 'openai_moderation', 'human_majority'}
        self.assertFalse(forbidden.intersection(FIELDS))

    def test_eligible_training_does_not_overlap_reserved_data(self):
        reserved = {r['duplicate_group'] for r in self.rows if r['dataset_role'] != 'training_candidate'}
        eligible = [r for r in self.rows if r['training_eligible'] == 'true']
        self.assertEqual(len(eligible), 4837)
        self.assertFalse({r['duplicate_group'] for r in eligible} & reserved)
        self.assertTrue(all(r['source_dataset'] == 'toxic_chat' and r['source_split'] == 'train'
                            and not r['quality_flags'] for r in eligible))

    def test_committed_outputs_reproduce(self):
        for name, content in build_artifacts(DATA).items():
            self.assertTrue((DATA / name).read_bytes() == content, name)

    def test_unexpected_schema_or_malformed_csv_fails(self):
        for content in ('wrong\nvalue\n', 'expected\na,b\n', 'expected,second\na\n'):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as archive:
                archive.writestr('test.csv', content)
            with zipfile.ZipFile(buffer) as archive:
                fields = ['expected', 'second'] if content.startswith('expected,') else ['expected']
                with self.assertRaises(ValueError):
                    read_member(archive, 'test.csv', fields)


if __name__ == '__main__':
    unittest.main()
