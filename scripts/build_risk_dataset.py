"""Build task 2 artifacts from the committed ZIPs using only the standard library."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import unicodedata
import zipfile

Record = dict[str, str]
Manifest = dict[str, object]

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    'record_id', 'source_dataset', 'source_config', 'source_split', 'source_file',
    'source_row', 'source_id', 'prompt', 'label', 'toxicity', 'jailbreaking',
    'human_annotation', 'behavior', 'category', 'behavior_source', 'dataset_role',
    'prompt_sha256', 'duplicate_group', 'duplicate_count', 'cross_partition_duplicate',
    'label_conflict', 'quality_flags', 'training_eligible', 'governance_action',
]
TOXIC_FIELDS = ['conv_id', 'user_input', 'model_output', 'human_annotation',
                'toxicity', 'jailbreaking', 'openai_moderation']
BEHAVIOR_FIELDS = ['Index', 'Goal', 'Target', 'Behavior', 'Category', 'Source']
JUDGE_FIELDS = ['Index', 'goal', 'prompt', 'target_response', 'human1', 'human2',
                'human3', 'human_majority', 'harmbench_cf', 'gpt4_cf',
                'llamaguard2_cf', 'llama3_cf']
ACTIONS = {'Safe': 'allow', 'Toxic': 'block', 'Jailbreak': 'block',
           'Harmful': 'block', 'Unknown': 'review'}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def classify_toxic_chat(toxicity: str, jailbreaking: str) -> str:
    """Keep explicit jailbreak evidence; incomplete negatives are not safe labels."""
    if toxicity not in ('', '0', '1') or jailbreaking not in ('', '0', '1'):
        raise ValueError('Unexpected Toxic Chat risk annotation')
    if jailbreaking == '1':
        return 'Jailbreak'
    if '' in (toxicity, jailbreaking):
        return 'Unknown'
    return 'Toxic' if toxicity == '1' else 'Safe'


def make_record(*, dataset: str, config: str, split: str, member: str,
                row_number: int, source_id: str, prompt: str, label: str) -> Record:
    if label not in ACTIONS:
        raise ValueError('Unexpected derived label')
    row = dict.fromkeys(FIELDS, '')
    row.update(record_id=f'{dataset}:{config}:{split}:{source_id}',
               source_dataset=dataset, source_config=config, source_split=split,
               source_file=member, source_row=str(row_number), source_id=source_id,
               prompt=prompt, label=label)
    if dataset == 'toxic_chat':
        row['dataset_role'] = 'training_candidate' if split == 'train' else 'held_out_test'
    else:
        row['dataset_role'] = 'judge_evaluation' if config == 'judge_comparison' else 'behavior_benchmark'
    return row


def read_member(archive: zipfile.ZipFile, member: str, fields: list[str]) -> list[Record]:
    with archive.open(member) as stream:
        reader = csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig', newline=''))
        if reader.fieldnames != fields:
            raise ValueError(f'Unexpected schema: {member}')
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f'Malformed CSV row: {member}')
    return rows


def load_records(data_dir: Path) -> tuple[list[Record], list[Record], Manifest]:
    records, judges, manifest = [], [], {}
    specs = {
        'lmsys_toxicchat.zip': {
            'lmsys_toxicchat/train.csv': TOXIC_FIELDS,
            'lmsys_toxicchat/test.csv': TOXIC_FIELDS,
        },
        'jailbreak.zip': {
            'jailbreak/behaviors_benign.csv': BEHAVIOR_FIELDS,
            'jailbreak/behaviors_harmful.csv': BEHAVIOR_FIELDS,
            'jailbreak/judge_comparison_test.csv': JUDGE_FIELDS,
        },
    }
    for archive_name, members in specs.items():
        archive_path = data_dir / archive_name
        manifest[archive_name] = {'sha256': digest(archive_path.read_bytes()), 'members': {}}
        with zipfile.ZipFile(archive_path) as archive:
            csv_members = [name for name in archive.namelist() if name.endswith('.csv')]
            if sorted(csv_members) != sorted(members):
                raise ValueError(f'Unexpected CSV members: {archive_name}')
            for member, fields in members.items():
                rows = read_member(archive, member, fields)
                manifest[archive_name]['members'][member] = {
                    'sha256': digest(archive.read(member)), 'rows': len(rows),
                    'columns': fields,
                    'empty_cells': {key: sum(not row[key].strip() for row in rows) for key in fields},
                }
                for number, source in enumerate(rows, 1):
                    if fields == TOXIC_FIELDS:
                        split = Path(member).stem
                        if source['human_annotation'] not in ('True', 'False', ''):
                            raise ValueError(f'Unexpected annotation provenance: {member}:{number}')
                        record = make_record(
                            dataset='toxic_chat', config='annotations', split=split, member=member,
                            row_number=number, source_id=source['conv_id'], prompt=source['user_input'],
                            label=classify_toxic_chat(source['toxicity'], source['jailbreaking']))
                        record.update({key: source[key] for key in ('toxicity', 'jailbreaking', 'human_annotation')})
                    elif fields == BEHAVIOR_FIELDS:
                        split = 'benign' if member.endswith('behaviors_benign.csv') else 'harmful'
                        record = make_record(
                            dataset='jailbreakbench', config='behaviors', split=split, member=member,
                            row_number=number, source_id=source['Index'], prompt=source['Goal'],
                            label='Safe' if split == 'benign' else 'Harmful')
                        record.update(behavior=source['Behavior'], category=source['Category'],
                                      behavior_source=source['Source'])
                    else:
                        record = make_record(
                            dataset='jailbreakbench', config='judge_comparison', split='test',
                            member=member, row_number=number, source_id=source['Index'],
                            prompt=source['prompt'], label='Unknown')
                        # Response-level judgments are never prompt-classifier labels/features.
                        judges.append({'record_id': record['record_id'], **source})
                    if not record['source_id'].strip():
                        raise ValueError(f'Missing source ID: {member}:{number}')
                    records.append(record)
    if len({row['record_id'] for row in records}) != len(records):
        raise ValueError('Duplicate source record IDs')
    return records, judges, manifest


def finalize_records(records: list[Record]) -> dict[str, list[Record]]:
    """Annotate quality/overlap without removing rows or changing source splits."""
    groups = defaultdict(list)
    for row in records:
        prompt = row['prompt']
        row['prompt_sha256'] = digest(prompt.encode('utf-8'))
        normalized = ' '.join(unicodedata.normalize('NFKC', prompt).casefold().split())
        row['duplicate_group'] = digest(normalized.encode('utf-8'))
        groups[row['duplicate_group']].append(row)
    for group in groups.values():
        partitions = {(r['source_dataset'], r['source_config'], r['source_split']) for r in group}
        known_labels = {r['label'] for r in group} - {'Unknown'}
        conflict = len(known_labels) > 1
        for row in group:
            flags = []
            if not row['prompt'].strip():
                flags.append('missing_prompt')
            if row['label'] == 'Unknown':
                flags.append('no_prompt_risk_label')
            if row['source_dataset'] == 'toxic_chat':
                if any(row[k] == '' for k in ('toxicity', 'jailbreaking', 'human_annotation')):
                    flags.append('missing_annotation')
                if row['jailbreaking'] == '1' and row['toxicity'] == '0':
                    flags.append('inconsistent_source_signals')
            if conflict:
                flags.append('conflicting_duplicate_labels')
            if len(partitions) > 1:
                flags.append('cross_partition_duplicate')
            row.update(duplicate_count=str(len(group)),
                       cross_partition_duplicate=str(len(partitions) > 1).lower(),
                       label_conflict=str(conflict).lower(), quality_flags=';'.join(flags),
                       training_eligible=str(row['dataset_role'] == 'training_candidate' and not flags).lower(),
                       governance_action='review' if any(f != 'cross_partition_duplicate' for f in flags)
                       else ACTIONS[row['label']])
    return groups


def csv_bytes(rows: list[Record], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


def build_artifacts(data_dir: Path) -> dict[str, bytes]:
    records, judges, manifest = load_records(data_dir)
    groups = finalize_records(records)
    counts = Counter((r['source_dataset'], r['source_split'], r['label']) for r in records)
    report = {
        'schema_version': 1,
        'sources': manifest,
        'total_rows': len(records),
        'label_counts': dict(sorted(Counter(r['label'] for r in records).items())),
        'counts_by_source_split_label': [
            {'source_dataset': key[0], 'source_split': key[1], 'label': key[2], 'count': value}
            for key, value in sorted(counts.items())
        ],
        'training_eligible_rows': sum(r['training_eligible'] == 'true' for r in records),
        'training_eligible_label_counts': dict(sorted(Counter(
            r['label'] for r in records if r['training_eligible'] == 'true').items())),
        'missing_prompt_rows': sum(not r['prompt'].strip() for r in records),
        'exact_duplicate_excess_rows': len(records) - len({r['prompt_sha256'] for r in records}),
        'normalized_duplicate_groups': sum(len(group) > 1 for group in groups.values()),
        'normalized_duplicate_excess_rows': len(records) - len(groups),
        'cross_partition_duplicate_rows': sum(r['cross_partition_duplicate'] == 'true' for r in records),
        'conflicting_label_rows': sum(r['label_conflict'] == 'true' for r in records),
        'flag_counts': dict(sorted(Counter(flag for r in records for flag in r['quality_flags'].split(';') if flag).items())),
    }
    # Full judge records remain separate; the combined table contains prompt-side fields only.
    artifacts = {
        'combined_risk_dataset.csv': csv_bytes(records, FIELDS),
        'judge_evaluation.csv': csv_bytes(judges, ['record_id'] + JUDGE_FIELDS),
    }
    report['artifact_sha256'] = {name: digest(content) for name, content in artifacts.items()}
    artifacts['risk_dataset_report.json'] = (json.dumps(report, indent=2, sort_keys=True) + '\n').encode('utf-8')
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--check', action='store_true', help='Verify committed artifacts without writing')
    args = parser.parse_args()
    artifacts = build_artifacts(args.data_dir)
    if args.check:
        stale = [name for name, content in artifacts.items()
                 if not (args.output_dir / name).is_file() or (args.output_dir / name).read_bytes() != content]
        if stale:
            parser.exit(1, 'Missing or stale artifacts: ' + ', '.join(stale) + '\n')
        print('All three artifacts reproduce byte-for-byte.')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in artifacts.items():
            (args.output_dir / name).write_bytes(content)
        print('Generated combined dataset, separate judge evaluation, and validation report.')


if __name__ == '__main__':
    main()
