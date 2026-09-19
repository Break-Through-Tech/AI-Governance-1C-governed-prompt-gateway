# Risk categories and governance rules

Task 2 implements the dataset combination, label mapping, and edge-case documentation requested in [issue #2](https://github.com/Break-Through-Tech/AI-Governance-1C-governed-prompt-gateway/issues/2). It does not train a model or implement the live gateway. Rules below are a versioned starting policy for the prototype, not measured classifier behavior or a guarantee of safety.

## Deliverables and reproduction

From the repository root, use Python 3.9 or later (standard library only):

```bash
python3 scripts/build_risk_dataset.py
python3 scripts/build_risk_dataset.py --check
python3 -m unittest discover -s tests -v
```

The generator reads the two committed ZIP archives without extracting or modifying them. No network requests, package installs, or API credentials are required. `--check` regenerates in memory and fails if any output differs. To inspect regenerated files separately, use `--output-dir /path/to/output`.

- [combined_risk_dataset.csv](../data/combined_risk_dataset.csv): one prompt-side table covering every row of all five source CSVs (10,665 rows).
- [judge_evaluation.csv](../data/judge_evaluation.csv): the 300 complete response-judging records, joined to the combined table by `record_id`. This file is for response evaluation only.
- [risk_dataset_report.json](../data/risk_dataset_report.json): source archive/member checksums, source schemas and empty-cell counts, derived label counts, duplicate diagnostics, training eligibility, and artifact checksums.
- [build_risk_dataset.py](../scripts/build_risk_dataset.py): deterministic transformation and quality flags.

The inputs are the repository's archive snapshot. Their exact upstream release/revision is not recorded in the archives; the generator identifies their contents by SHA-256 rather than assuming a Toxic Chat version. Source files, row order, prompt text, and original split names are preserved. `source_row` is the one-based data-row number, excluding the header.

## Taxonomy and source mapping

The issue suggests Safe/Toxic/Jailbreak. We retain those classes and add **Harmful** to avoid mislabeling harmful goals as jailbreak attacks. **Unknown** is an abstention status, not a supervised target class.

| Label | Evidence used for this artifact | Planned gateway action |
|---|---|---|
| Safe | Toxic Chat `toxicity=0` and `jailbreaking=0`, or a JBB benign behavior | Allow to proceed to subsequent checks |
| Toxic | Toxic Chat `toxicity=1` and `jailbreaking=0` | Block direct execution; offer a safe redirection |
| Jailbreak | Explicit Toxic Chat `jailbreaking=1`, including overlap with toxicity | Block direct execution; offer a safe redirection |
| Harmful | A JBB harmful behavior goal; no claim about attack technique | Block direct execution; offer a safe redirection |
| Unknown | Missing prompt-risk evidence, including all judge-comparison records | Review; do not execute automatically |

These classes describe source evidence, not mutually exclusive real-world concepts. `Jailbreak` takes precedence over `Toxic` in the single `label` column; both original signals remain available. A source-negative `Safe` label means no annotated risk under that source's coverage, not that the request satisfies every future domain policy.

The [Toxic Chat dataset card](https://huggingface.co/datasets/lmsys/toxic-chat) defines separate toxicity and jailbreaking annotations. Its [paper, section 2.2](https://arxiv.org/html/2310.17389v1) explains the human/AI annotation process: `human_annotation=False` does not mean that the released risk labels are missing. Those labels are retained, and human annotation provenance remains visible for evaluation by subgroup.

The [JBB dataset card](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) distinguishes behavior goals, affirmative response targets, and the separate judge-comparison configuration. The upstream [judge implementation](https://github.com/JailbreakBench/jailbreakbench/blob/main/src/jailbreakbench/classifier.py) evaluates responses. A response judge's positive or negative result cannot be used as the input prompt's risk class. We do not infer Jailbreak from the archive name, harmful split, or response outcome.

## Combined-table contract

| Columns | Meaning and downstream use |
|---|---|
| `record_id` | Unique dataset/configuration/split/source-ID key; joins judge records |
| `source_dataset`, `source_config`, `source_split`, `source_file`, `source_row`, `source_id` | Provenance; source configuration `annotations` is a local descriptor, not an asserted Toxic Chat release |
| `prompt` | **Only input feature for the prompt classifier:** Toxic Chat `user_input`, JBB behavior `Goal`, or judge-comparison `prompt` |
| `label` | Derived source-backed label; Unknown is excluded from training |
| `toxicity`, `jailbreaking`, `human_annotation` | Original Toxic Chat signals, unchanged; empty for JBB |
| `behavior`, `category`, `behavior_source` | Original JBB behavior metadata; empty where not applicable |
| `dataset_role` | `training_candidate`, `held_out_test`, `behavior_benchmark`, or `judge_evaluation` |
| `prompt_sha256` | Exact UTF-8 prompt digest |
| `duplicate_group`, `duplicate_count` | Digest/group size after Unicode NFKC normalization, case folding, and whitespace collapse; original prompt is unchanged |
| `cross_partition_duplicate` | Normalized prompt occurs in more than one dataset/configuration/split partition |
| `label_conflict` | Normalized duplicate group has more than one known label; Unknown is not treated as contradictory evidence |
| `quality_flags` | Semicolon-delimited review/eligibility reasons; empty means none detected |
| `training_eligible` | `true` only for Toxic Chat training rows without quality flags |
| `governance_action` | Proposed `allow`, `block`, or `review` policy action; not a model prediction |

Boolean flags use lowercase `true`/`false`; original `human_annotation` retains `True`/`False`. Empty source-specific cells mean unavailable/not applicable, not a negative annotation.

Do not vectorize the entire table. Labels, category, source identity, annotation signals, and derived actions/flags would leak supervision or source shortcuts into model features. Toxic Chat `model_output`/`openai_moderation` and JBB `Target` are deliberately absent from the combined table and remain in the original ZIPs. Judge responses and judgments are isolated in the evaluation file; do not join them into classifier features. A request may itself contain quoted conversation text; the generator preserves that original input rather than adding response columns to it.

## Split preservation and downstream handoff

- Toxic Chat's original `train` and `test` remain intact. Do not randomly resplit the combined table. Only `training_eligible=true` rows are current training candidates.
- Task 5 can create a validation set from eligible training rows, grouping by `duplicate_group` so repeated prompts stay together. Fit vocabulary, vectorization, resampling, and other learned transformations on training data only.
- JBB `benign` and `harmful` are behavior partitions, not train/test splits. This implementation conservatively reserves both as a behavior benchmark. Using them for training would require an explicit new evaluation protocol; the same behaviors could no longer be claimed as unseen tests.
- The judge-comparison `test` partition is exclusively response evaluation. All its combined-table labels are Unknown, and none are training-eligible.
- Normalized cross-partition duplicates are excluded from **training eligibility**, not deleted. Official test rows are retained. Report metrics with the source split, human-annotation subgroup, and overlap handling stated explicitly.
- Same-partition duplicates are retained and flagged by group size. Deduplication/reweighting belongs to the cleanup task; this change does not silently select rows or alter class balance.
- Matching detects exact and normalized text reuse, not semantic paraphrases or shared behavior goals hidden inside longer attack prompts. Benchmark isolation remains necessary even when hashes differ.

**Task 6 implication:** eligible training data has Safe/Toxic/Jailbreak examples only. There are no eligible Harmful training examples in this snapshot. Train/evaluate the supported three-class baseline separately; do not claim a learned four-class classifier or fold Harmful into Jailbreak. Harmful remains an explicit benchmark/policy distinction pending additional labeled training data and a revised protocol.

## Edge cases and governance behavior

| Case | Mapping and handling |
|---|---|
| Both toxicity and jailbreak are positive | Label Jailbreak; retain both source flags; block |
| Explicit jailbreak positive but toxicity negative | Label Jailbreak; flag inconsistent source signals; review and exclude from training pending adjudication |
| Jailbreak positive with missing toxicity | Retain Jailbreak evidence; flag missing annotation; review and exclude from training |
| Missing toxicity or jailbreak without explicit jailbreak evidence | Unknown; review; never assume Safe |
| Invalid risk encoding, unexpected CSV schema/members, malformed rows, or missing/duplicate source IDs | Fail generation with an error instead of silently changing the mapping |
| Human annotation is False | Retain the released labels and provenance; do not confuse automated annotation with missing annotation |
| Human annotation field is missing | Retain available evidence but flag and exclude from training; review |
| Harmful goal with no jailbreak annotation | Harmful; block; do not manufacture a Jailbreak label |
| Benign and harmful behaviors share the same category name | Use the source behavior partition, not the category name, for the mapping |
| Model refuses a harmful request | Refusal is a response outcome; it does not make the prompt Safe |
| Judge votes disagree or a judge reports success | Retain votes in the separate evaluation artifact; prompt label stays Unknown |
| Prompt is empty or whitespace-only | Preserve the record and label evidence; flag missing prompt, review, exclude from training |
| Normalized duplicate prompts have conflicting known labels | Preserve labels; flag conflict across the whole group; review and exclude from training until adjudicated |
| Prompt repeats across partitions | Preserve source splits; flag overlap; exclude affected training rows; overlap alone does not change risk action |
| Quoted harmful text, educational discussion, roleplay, or unclear intent | No keyword-based relabeling. Retain source labels; future uncertain live predictions should go to review |
| Missing domain coverage or low-confidence live prediction | Review without execution; confidence thresholds and domain policies are To be determined |

Review means abstain from automatic execution until resolved, not silently allow. Blocked/reviewed prompts should not enter optimization, cache reuse, or LLM execution before that decision is resolved. Future logging should capture the policy version, decision, and reason; the dataset's actions are policy defaults, not an implemented gateway or confidence model.

## Snapshot validation findings

| Source partition | Safe | Toxic | Jailbreak | Harmful | Unknown | Total |
|---|---:|---:|---:|---:|---:|---:|
| Toxic Chat train | 4,698 | 271 | 113 | 0 | 0 | 5,082 |
| Toxic Chat test | 4,721 | 271 | 91 | 0 | 0 | 5,083 |
| JBB benign | 100 | 0 | 0 | 0 | 0 | 100 |
| JBB harmful | 0 | 0 | 0 | 100 | 0 | 100 |
| JBB judge comparison test | 0 | 0 | 0 | 0 | 300 | 300 |
| **Total** | **9,519** | **542** | **204** | **100** | **300** | **10,665** |

The committed report records zero missing prompts and zero conflicting known labels within normalized duplicate groups. There are 398 exact duplicate excess rows, and 480 normalized duplicate excess rows across 439 duplicate groups. “Excess” means total rows minus unique prompt groups; it is not the number of all rows involved in duplicates. Across partitions, 495 rows are flagged. After the conservative eligibility filter, 4,837 Toxic Chat training rows remain: 4,488 Safe, 254 Toxic, and 95 Jailbreak. These are dataset diagnostics, not model scores.

## Attribution and limitations

Sources: LMSYS Toxic Chat, *ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation* (Lin et al., 2023); JailbreakBench, *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models* (Chao et al., 2024). Source cards and the Toxic Chat paper are linked above. JBB behavior `Source` attribution is retained per row.

Toxic Chat text is distributed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); JBB is distributed under [MIT](https://github.com/JailbreakBench/jailbreakbench/blob/main/LICENSE). This derived dataset does not replace those terms or establish a license for the rest of this project. The transformation adds labels, provenance and quality flags and omits response-side fields from the combined table; it does not rewrite prompt text.

No prompt examples are reproduced here. The generated CSVs contain the same potentially offensive source text already in the archives. Exact source revisions, domain-policy calibration, adjudication of future conflicts, and validation of a trained model remain To be determined.
