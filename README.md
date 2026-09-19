# Governed Prompt Gateway

Break Through Tech AI Studio · Fall 2026 · AI Governance 1C

---

### 👥 **Team Members**

These responsibilities reflect September assignments, not completed contributions.

| Name | Planned responsibility |
|------|------------------------|
| Umar Faheem | Dataset loading, exploratory data analysis (EDA), and data cleanup |
| Prince Osei Boakye | Risk categories and governance rules; SinhSinh An is stepping in as substitute |
| SinhSinh An | Risk categories and governance rules as substitute; token usage and cost baseline |
| Manuel Arellano | System architecture and end-to-end data flow |
| Linda Chen | Dataset splitting and text vectorization |
| Angelina Kovalchuk | Baseline classifier development |
| Barsat Khadka | Model evaluation |

---

## 🎯 **Project Highlights**

- The project aims to build middleware that screens prompt intent and risk before LLM execution, blocks or redirects unsafe requests, and optimizes eligible prompts.
- The planned workflow includes semantic caching, LLM-tier recommendations, a dashboard, and auditable decision logs.
- Success targets are at least 80% F1 for prompt-risk classification and 30–50% input-token reduction for eligible prompts, alongside measured cache-hit rate and simulated cost savings. These are targets, not achieved results.
- Achieved model and gateway performance: To be determined

---

## 👩🏽‍💻 **Setup and Installation**

1. Clone the repository using Git:

   ```bash
   git clone https://github.com/Break-Through-Tech/AI-Governance-1C-governed-prompt-gateway.git
   cd AI-Governance-1C-governed-prompt-gateway
   ```

2. Access the dataset archives included in [data/](data/):
   - [lmsys_toxicchat.zip](data/lmsys_toxicchat.zip)
   - [jailbreak.zip](data/jailbreak.zip)

   Extract each archive into its own directory for inspection. Source dataset cards are linked under References.

3. Dependency installation: To be determined

   The current [requirements.txt](requirements.txt) is empty.

4. Environment configuration: To be determined

5. Notebook and application execution instructions: To be determined

   `main` does not yet contain a runnable notebook or gateway application. A preliminary EDA notebook is available on the [Task-#1-and-4--Initial-EDA branch](https://github.com/Break-Through-Tech/AI-Governance-1C-governed-prompt-gateway/blob/Task-%231-and-4--Initial-EDA/notebooks/01_eda.ipynb); its environment and reproduction instructions have not been finalized.

---

## 🏗️ **Project Overview**

Governed Prompt Gateway is the AI Governance 1C team's Fall 2026 Break Through Tech AI Studio project. It explores how a governance layer can make LLM use safer, more token-efficient, and easier to audit for regulated customer-service scenarios.

The planned gateway classifies a query's intent and risk, blocks or redirects unsafe requests, optimizes eligible prompts, checks for semantically similar cached queries, and recommends an LLM tier. A dashboard will display governance decisions, token savings, cost estimates, cache status, and model recommendations. The final response may be generated or simulated, with an auditable decision log.

This is a student prototype project; an integrated gateway has not yet been delivered. The [challenge overview](Challenge-Project-Overview.md) defines the scope and success criteria.

Host company: To be determined

---

## 📊 **Data Exploration**

The repository contains CSV datasets packaged in two ZIP archives:

| Dataset | Contents of the repository archive | Intended use |
|---------|------------------------------------|--------------|
| LMSYS Toxic Chat | `train.csv` (5,082 rows) and `test.csv` (5,083 rows), including user inputs, model outputs, toxicity and jailbreaking annotations | Prompt-risk exploration and baseline classification |
| JailbreakBench | `behaviors_benign.csv` (100 rows), `behaviors_harmful.csv` (100 rows), and `judge_comparison_test.csv` (300 rows with a separate evaluation schema) | Explore benign and harmful behaviors and benchmark context |

Initial EDA work is recorded in the separate notebook branch linked above. September assignments cover label distributions, prompt lengths, duplicates, missing values, class imbalance, and data cleanup.

The proposed Safe/Toxic/Jailbreak taxonomy is still being reconciled with the source annotations. Harmful behavior is not automatically a jailbreak attempt, and the judge-comparison file is not interchangeable with the behavior tables. Source evaluation boundaries must be preserved when finalizing the training and evaluation data.

- Final preprocessing and label mapping: To be determined
- Validated EDA findings and visualizations: To be determined

---

## 🧠 **Model Development**

The September baseline plan is to vectorize prompt text with TF-IDF and compare candidate classifiers: K-nearest neighbors, logistic regression, support vector machines, and Naive Bayes. Evaluation will report accuracy, precision, recall, and F1.

- Selected model and hyperparameters: To be determined
- Final training, validation, and test split: To be determined
- Trained baseline and performance: To be determined
- Tokenizer and cost-estimation model: To be determined
- Final implementation stack: To be determined

The planned gateway extends this baseline with governance rules, prompt optimization, semantic caching, and model-tier recommendations.

---

## 📈 **Results & Key Findings**

The project is in the data exploration and baseline-development stage. No trained-classifier scores or integrated-gateway evaluation results are available on `main`.

| Measure | Project target or evaluation plan | Measured result |
|---------|-----------------------------------|-----------------|
| Prompt-risk classification | At least 80% F1; also report accuracy, precision, and recall | To be determined |
| Input-token reduction | 30–50% for eligible prompts | To be determined |
| Semantic cache effectiveness | Measure cache-hit rate for repeated and similar queries | To be determined |
| Cost efficiency | Estimate savings on a simulated customer-service workload | To be determined |

- Fairness evaluation findings: To be determined
- Evaluation visualizations: To be determined

---

## 🚀 **Next Steps**

The confirmed milestone plan is:

- **September:** Finalize scope and risk categories, explore and clean data, define the architecture, compare baseline classifiers, and establish token and cost baselines.
- **October:** Build the governance layer, prompt optimization, and semantic caching prototype.
- **November:** Integrate the workflow, evaluate the system, add the dashboard and model recommendations, and prepare the presentation and demo.
- **December:** Demonstrate the governed prompt gateway with generated or simulated responses and auditable decisions.

Current open work is tracked in the [repository issues](https://github.com/Break-Through-Tech/AI-Governance-1C-governed-prompt-gateway/issues).

- Measured model limitations: To be determined
- Additional datasets or techniques beyond the confirmed scope: To be determined
- Changes with more time or resources: To be determined

---

## 📝 **License**

Project license: To be determined

The repository does not currently specify a project license. Dataset usage terms are documented by their respective sources.

---

## 📄 **References** (Optional but encouraged)

- [Challenge project overview](Challenge-Project-Overview.md)
- [Getting started for fellows](Getting-Started-for-Fellows.md)
- [September task plan](https://docs.google.com/document/d/1KOQ19jmjarUj3OPsK3CMHMozqT6SxvEvGo-ASGw0S8s/edit)
- [Team project brief](https://docs.google.com/document/d/1-EuB3O0gyC6zpnodWO1kv5fCFxfVMZuFbWw60xpSGl4/edit)
- [LMSYS Toxic Chat dataset and documentation](https://huggingface.co/datasets/lmsys/toxic-chat)
- [JailbreakBench JBB-Behaviors dataset and documentation](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)

Team documents may require access permissions.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank you to our Challenge Advisor, **Bhavana Unnam**, and AI Studio Coach, **Om Kamath**, for their guidance, and to **Break Through Tech AI Studio** for supporting this project.
