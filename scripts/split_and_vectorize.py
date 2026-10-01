"""Prepare the cleaned dataset for baseline classification (issue #5)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import unicodedata

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'safe', 'toxic', 'jailbreak'}
FIELDS = ['user_input', 'label']


def prompt_group(row):
    """Match prompts without changing the cleaned text used for features."""
    return ' '.join(unicodedata.normalize('NFKC', row['user_input']).casefold().split())


def read_cleaned(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f'{path}: expected user_input,label columns')
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f'{path}: malformed CSV row')
    return rows


def split_records(train_rows, test_rows, validation_size=0.15, seed=42):
    """Split cleaned training groups and preserve the cleaned test set."""
    if not 0 < validation_size < 1:
        raise ValueError('validation_size must be between 0 and 1')
    for rows in (train_rows, test_rows):
        if not rows or any(set(row) != set(FIELDS) for row in rows):
            raise ValueError('Expected nonempty cleaned data with user_input,label columns')
        if any(r['label'] not in LABELS or not prompt_group(r) for r in rows):
            raise ValueError('Cleaned rows need a nonempty prompt and a supported label')

    # Cleanup removed exact overlap, but case/whitespace variants remain.
    test_groups = {prompt_group(row) for row in test_rows}
    eligible = [r for r in train_rows if prompt_group(r) not in test_groups]
    groups = defaultdict(list)
    for row in eligible:
        groups[prompt_group(row)].append(row)
    if any(len({r['label'] for r in group}) != 1 for group in groups.values()):
        raise ValueError('Duplicate prompt group has conflicting labels')

    group_ids = sorted(groups)
    group_labels = [groups[key][0]['label'] for key in group_ids]
    if set(group_labels) != LABELS:
        raise ValueError('Cleaned training data must contain all three baseline classes')
    try:
        train_ids, val_ids = train_test_split(
            group_ids, test_size=validation_size, random_state=seed,
            stratify=group_labels,
        )
    except ValueError as error:
        raise ValueError('Not enough prompt groups for a stratified split of this size') from error

    train_ids, val_ids = set(train_ids), set(val_ids)
    splits = {
        'train': [r for r in eligible if prompt_group(r) in train_ids],
        'val': [r for r in eligible if prompt_group(r) in val_ids],
        'test': test_rows,
    }
    if any({r['label'] for r in splits[name]} != LABELS for name in ('train', 'val')):
        raise ValueError('Split would leave a class out of training or validation')
    return splits


def vectorize(splits):
    """Learn vocabulary and IDF from training prompts only."""
    vectorizer = TfidfVectorizer(
        max_features=10000, ngram_range=(1, 2), sublinear_tf=True,
    )
    matrices = {'train': vectorizer.fit_transform([r['user_input'] for r in splits['train']])}
    for name in ('val', 'test'):
        matrices[name] = vectorizer.transform([r['user_input'] for r in splits[name]])
    return vectorizer, matrices


def prepare_dataset(train_path, test_path, output_dir, validation_size=0.15, seed=42):
    train_path, test_path, output_dir = Path(train_path), Path(test_path), Path(output_dir)
    train_rows, test_rows = read_cleaned(train_path), read_cleaned(test_path)
    splits = split_records(train_rows, test_rows, validation_size, seed)
    vectorizer, matrices = vectorize(splits)
    labels = {name: np.array([r['label'] for r in records]) for name, records in splits.items()}
    classes = np.unique(labels['train'])
    weights = compute_class_weight('balanced', classes=classes, y=labels['train'])
    class_weights = dict(zip(classes.tolist(), weights.tolist()))
    report = {
        'inputs': {name: {'filename': path.name,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in [('train', train_path), ('test', test_path)]},
        'seed': seed,
        'validation_fraction_of_training_groups': validation_size,
        'protocol': 'Split cleaned training groups; preserve cleaned test rows.',
        'excluded_training_overlap_rows': len(train_rows) - len(splits['train']) - len(splits['val']),
        'vocabulary_size': len(vectorizer.vocabulary_),
        'tfidf': {'max_features': 10000, 'ngram_range': [1, 2], 'sublinear_tf': True},
        'versions': {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'joblib')},
        'class_weights': class_weights,
        'splits': {
            name: {
                'rows': len(records),
                'groups': len({prompt_group(r) for r in records}),
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
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
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
    parser.add_argument('--train-input', type=Path, default=ROOT / 'data/train_clean.csv')
    parser.add_argument('--test-input', type=Path, default=ROOT / 'data/test_clean.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed/tfidf')
    parser.add_argument('--validation-size', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    report = prepare_dataset(args.train_input, args.test_input, args.output_dir,
                             args.validation_size, args.seed)
    for name, split in report['splits'].items():
        print(f"{name}: {split['rows']} rows, {split['labels']}, shape={split['shape']}")
    print(f'Saved splits, features, labels, and fitted vectorizer to {args.output_dir}')


if __name__ == '__main__':
    main()
