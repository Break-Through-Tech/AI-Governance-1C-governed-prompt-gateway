"""Prepare the task 2 dataset for baseline classification (issue #5)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'Safe', 'Toxic', 'Jailbreak'}
REQUIRED_COLUMNS = {
    'record_id', 'prompt', 'label', 'source_dataset', 'source_split',
    'dataset_role', 'duplicate_group', 'training_eligible', 'quality_flags',
}


def split_records(rows, validation_size=0.15, seed=42):
    """Stratify eligible prompt groups; preserve the official Toxic Chat test set."""
    if not 0 < validation_size < 1:
        raise ValueError('validation_size must be between 0 and 1')
    if not rows or any(not REQUIRED_COLUMNS <= row.keys() for row in rows):
        raise ValueError('Expected the combined dataset produced by task 2')
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate record IDs in the input dataset')

    eligible = [r for r in rows if r['training_eligible'] == 'true']
    test = [r for r in rows if r['source_dataset'] == 'toxic_chat'
            and r['source_split'] == 'test']
    if not eligible or not test:
        raise ValueError('Expected eligible training rows and the original test set')
    if any(r['source_dataset'] != 'toxic_chat' or r['source_split'] != 'train'
           or r['dataset_role'] != 'training_candidate' or r['quality_flags']
           for r in eligible):
        raise ValueError('Training eligibility conflicts with source provenance or quality flags')
    if any(r['label'] not in LABELS or not r['prompt'].strip()
           or not r['duplicate_group'] for r in eligible + test):
        raise ValueError('Training and test rows need a prompt, group, and supported label')

    reserved_groups = {r['duplicate_group'] for r in rows
                       if r['dataset_role'] != 'training_candidate'}
    groups = defaultdict(list)
    for row in eligible:
        groups[row['duplicate_group']].append(row)
    if set(groups) & reserved_groups:
        raise ValueError('Eligible training prompts overlap reserved evaluation data')
    if any(len({r['label'] for r in group}) != 1 for group in groups.values()):
        raise ValueError('Duplicate prompt group has conflicting labels')

    # One item per group makes stratification compatible with duplicate isolation.
    group_ids = sorted(groups)
    group_labels = [groups[key][0]['label'] for key in group_ids]
    if set(group_labels) != LABELS:
        raise ValueError('Eligible training data must contain all three baseline classes')
    try:
        train_ids, val_ids = train_test_split(
            group_ids, test_size=validation_size, random_state=seed,
            stratify=group_labels,
        )
    except ValueError as error:
        raise ValueError('Not enough prompt groups for a stratified split of this size') from error

    train_ids, val_ids = set(train_ids), set(val_ids)
    splits = {
        'train': [r for r in eligible if r['duplicate_group'] in train_ids],
        'val': [r for r in eligible if r['duplicate_group'] in val_ids],
        'test': test,
    }
    if any({r['label'] for r in splits[name]} != LABELS for name in ('train', 'val')):
        raise ValueError('Split would leave a class out of training or validation')
    return splits


def vectorize(splits):
    """Learn vocabulary and IDF from training prompts only."""
    vectorizer = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2), sublinear_tf=True,
    )
    matrices = {'train': vectorizer.fit_transform([r['prompt'] for r in splits['train']])}
    for name in ('val', 'test'):
        matrices[name] = vectorizer.transform([r['prompt'] for r in splits[name]])
    return vectorizer, matrices


def prepare_dataset(input_path, output_dir, validation_size=0.15, seed=42):
    input_path, output_dir = Path(input_path), Path(output_dir)
    with input_path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError('Malformed row in the combined dataset')
    splits = split_records(rows, validation_size, seed)
    vectorizer, matrices = vectorize(splits)
    labels = {name: np.array([r['label'] for r in records]) for name, records in splits.items()}
    classes = np.unique(labels['train'])
    weights = compute_class_weight('balanced', classes=classes, y=labels['train'])
    class_weights = dict(zip(classes.tolist(), weights.tolist()))
    report = {
        'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'seed': seed,
        'validation_fraction_of_training_groups': validation_size,
        'protocol': 'Split eligible Toxic Chat training groups; preserve original test rows.',
        'excluded_rows': len(rows) - sum(len(records) for records in splits.values()),
        'vocabulary_size': len(vectorizer.vocabulary_),
        'tfidf': {'max_features': 10000, 'ngram_range': [1, 2], 'sublinear_tf': True},
        'versions': {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'joblib')},
        'class_weights': class_weights,
        'splits': {
            name: {
                'rows': len(records),
                'groups': len({r['duplicate_group'] for r in records}),
                'labels': dict(sorted(Counter(r['label'] for r in records).items())),
                'shape': list(matrices[name].shape),
                'zero_feature_rows': int(np.count_nonzero(matrices[name].getnnz(axis=1) == 0)),
            }
            for name, records in splits.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        # CSV rows, sparse matrix rows, and label-array positions have identical order.
        with (output_dir / f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(records)
        sparse.save_npz(output_dir / f'X_{name}.npz', matrices[name])
        np.save(output_dir / f'y_{name}.npy', labels[name], allow_pickle=False)
    joblib.dump(vectorizer, output_dir / 'tfidf_vectorizer.joblib')
    for filename, content in [('class_weights.json', class_weights), ('split_report.json', report)]:
        (output_dir / filename).write_text(json.dumps(content, indent=2, sort_keys=True) + '\n',
                                          encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/combined_risk_dataset.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/tfidf')
    parser.add_argument('--validation-size', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    report = prepare_dataset(args.input, args.output_dir, args.validation_size, args.seed)
    for name, split in report['splits'].items():
        print(f"{name}: {split['rows']} rows, {split['labels']}, shape={split['shape']}")
    print(f'Saved splits, features, labels, and fitted vectorizer to {args.output_dir}')


if __name__ == '__main__':
    main()
