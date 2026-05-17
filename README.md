# SLITE: Semantic Layers for Interpretable Textual Entailment

This repository contains the official implementation of **SLITE**, a hybrid, intrinsically explainable model for Recognizing Textual Entailment (RTE) / Natural Language Inference (NLI). SLITE integrates complementary layers of symbolic analysis and distributional semantics, proving that it is possible to achieve near-State-of-the-Art (SOTA) performance without sacrificing linguistic transparency or relying on opaque black-box deep learning architectures.

---

## 🚀 Key Features

* **Inherently Interpretable:** Unlike *post-hoc* explainability methods that merely approximate model behavior, SLITE is transparent by design, utilizing a logistic regression classifier over explicit, linguistically-grounded features.
* **Non-Serialized Architecture:** Eliminates the rigidity and cascading error propagation typical of traditional pipelines that rely strictly on Subject-Verb-Object (SVO) triplet extraction.
* **Symbolic-Distributional Fusion:** Leverages **ConceptNet** and its dense vector embeddings (**Numberbatch**) to unify logical-semantic relation extraction with geometric and information-theoretic analysis.
* **Computational Efficiency:** Achieves highly competitive results on standard benchmarks using a negligible fraction of the computational footprint and parameter count required by deep models like RoBERTa.

---

## 🧠 Model Architecture

SLITE operates by extracting **17 handcrafted features** structured into two complementary analytical dimensions from the Premise ($P$) and the Hypothesis ($H$):

### 1. Structural-Relational Layer (E-Features)
Using syntactic dependency parsing via *spaCy*, the model decomposes sentences into core compositional entities (nouns and verbs) and their modifiers (adjectives, adverbs). Elements in $H$ are mapped and classified into four mutually exclusive semantic groups relative to $P$ using the ConceptNet ontology:
* **Equivalence/Generality ($\mathcal{G}_{eq}$):** Synonyms and hypernyms.
* **Opposition ($\mathcal{G}_{op}$):** Antonyms and mutual exclusions.
* **Specificity ($\mathcal{G}_{sp}$):** Hyponyms.
* **Non-Relation ($\mathcal{G}_{nr}$):** Concepts lacking direct semantic alignment.

Additionally, this layer incorporates a **semantically-extended Jaro distance** metric that directionally penalizes misalignments, alongside polarity-sensitive components to robustly handle negation.

### 2. Distributional-Informational Layer (L-Features)
Moving beyond basic global cosine similarity, this layer constructs specific entity-to-lexicon similarity sub-matrices. By selectively masking rows and columns corresponding to certain semantic groups, the model measures residual textual coherence. Metrics rooted in Information Theory are applied to capture the directional information flow from $P \rightarrow H$:
* **Cosine Similarity Entropy:** Measures the uncertainty over the alignment similarity distributions.
* **Transfer Entropy:** Quantifies the directional reduction of uncertainty between the distributional representations of the two sentences.

---

## 📊 Experimental Results

SLITE was evaluated on the standard **SICK** (Sentences Involving Compositional Knowledge) dataset and its binary classification variant, **SICK-CE**.

| Model | SICK (3-Class) | SICK-CE (Binary) | Model Type |
| :--- | :---: | :---: | :--- |
| **SLITE** (Logistic Regression) | **83.0%** | **96.0%** | Hybrid / Intrinsically Interpretable |
| IsoLex (Souza & Lopes, 2025) | 79.0% | 92.0% | SVO-Based Hybrid / WordNet |
| RoBERTa (SOTA Opaque) | ~85.0% | 98.0% | Deep Neural Network / Black-Box |

### Explainability Insights (SHAP Analysis)
* **Classification Drivers:** Global SHAP values confirm that **Structural-Relational (E-Features)** dominate overall classification decisions.
* **Most Predictive Metrics:** The *lexico-semantic match ratio* and the *entailment weighting score* stand out as variables with the highest inferential weight.
* **Contradiction Detection:** **Distributional-Informational (L-Features)** provide critical complementary support, proving essential for identifying neutral and contradiction states by capturing subtle residual patterns of information loss.

---

## 🛠️ Requirements

* Python 3.8+
* spaCy (`en_core_web_md` model)
* ConceptNet Numberbatch Embeddings
* Scikit-Learn
* SHAP

---

## ✒️ Citation / Authors

* **David Torres-Moreno** - *Centro de Investigación en Ciencias, UAEM*
* **Jorge Hermosillo-Valadez** - *Centro de Investigación en Ciencias, UAEM*
* **Asela Reig-Alamillo** - *Centro Interdisciplinary de Investigación en Humanidades, UAEM*

---

## 🙏 Acknowledgments
This research was supported by a fellowship and funding from **CONACYT MÉXICO**.