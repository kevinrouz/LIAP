# Linguistically-Informed Attention Pruning for Interpretable NER

**Author:** Kevin Farokhrouz<br>
**Affiliation:** Junior CS Student, University of Texas at Arlington<br>
**Project Origin:** Kenny Zhu's NLP course, Spring 2025

---

## Table of Contents

- [Background \& Motivation](#background--motivation)
- [Approach](#approach)
    - [Model \& Dataset](#model--dataset)
    - [Linguistically-Informed Attention Pruning (LIAP)](#linguistically-informed-attention-pruning-liap)
    - [Baselines: LIME](#baselines-lime)
    - [Linguistic Faithfulness Score (LFS)](#linguistic-faithfulness-score-lfs)
- [Results](#results)
    - [Quantitative Analysis](#quantitative-analysis)
    - [Qualitative Examples](#qualitative-examples)
    - [Visualizations](#visualizations)
- [How to Run](#how-to-run)
- [Limitations \& Future Work](#limitations--future-work)
- [References](#references)

---

## Background \& Motivation

This project began as part of a research initiative in Kenny Zhu's NLP course at UTA in Spring 2025. The initial goal was to explore how transformer attention can be interpreted for NER models. As the research evolved, the focus shifted to a more linguistically grounded approach: **Linguistically-Informed Attention Pruning (LIAP)**. The central question became:
*How close are state-of-the-art NER models' attention patterns to our linguistic intuitions, and can we quantify and improve this alignment?*

---

## Approach

### Model \& Dataset

- **Model:** `bert-base-cased`
(Cased model chosen as NER tasks benefit from case sensitivity for names and locations.)
- **Dataset:** [CoNLL-2003 English NER](https://www.clips.uantwerpen.be/conll2003/ner/)
- **Fine-tuning:** Standard supervised fine-tuning on the CoNLL-2003 train/validation split. (See code for details.)


### Linguistically-Informed Attention Pruning (LIAP)

**LIAP** is a method for refining transformer attention maps using explicit linguistic knowledge, making explanations more faithful to how humans understand language.

- **Syntactic Dependency Pruning:**
Uses [spaCy](https://spacy.io/) to parse sentences. Prunes attention weights that do not align with dependency relations involving the entity, its head, and its dependents.
- **Semantic Role Labeling (SRL) Pruning:**
Uses [AllenNLP's SRL model](https://demo.allennlp.org/semantic-role-labeling) to identify semantic frames. Prunes attention to words not sharing a semantic role with the entity.

### Baselines: LIME

- **LIME (Local Interpretable Model-agnostic Explanations):**
Used as a baseline interpretability method for comparison with LIAP. See [`lime_baseline.py`](lime_baseline.py) for implementation.


### Linguistic Faithfulness Score (LFS)

- **Metric:** Perturbation-based LFS
For each example, important and unimportant tokens (as determined by pruned attention) are perturbed (masked), and the impact on the model's prediction is measured.
- **Continuous LFS:**
The score is continuous, reflecting the proportion of prediction changes caused by perturbing important tokens versus unimportant tokens.
    - LFS closer to 1: LIAP-pruned tokens are highly influential.
    - LFS near 0.5: Model is robust to perturbations (neutral).
    - LFS closer to 0: Unimportant tokens are as or more influential than important ones.

---

## Results

### Quantitative Analysis

| Metric | Value |
| :-- | :-- |
| Examples | 40 |
| Avg. LFS | 0.6626 |
| Median LFS | 0.5958 |
| Std. Dev. | 0.2386 |

**LFS by Entity Type:**


| Entity Type | Mean | Median | Std | Count |
| :-- | :-- | :-- | :-- | :-- |
| LOC | 0.572 | 0.557 | 0.244 | 10 |
| MISC | 0.744 | 0.817 | 0.243 | 9 |
| ORG | 0.691 | 0.797 | 0.240 | 7 |
| PER | 0.661 | 0.550 | 0.234 | 10 |

**LIAP vs LIME Comparison:**


| has_lime | Mean | Median | Std | Count |
| :-- | :-- | :-- | :-- | :-- |
| False | 0.632 | 0.571 | 0.231 | 30 |
| True | 0.818 | 0.890 | 0.235 | 6 |

### Qualitative Examples

**Top 5 Examples by LFS:**

- *Example 9 (PER, Hassan):* LFS 1.0000
_"Defender Hassan Abbas rose to intercept a long ball into the area in the 84th minute but only managed to divert it into the top corner of Bitar's goal."_
- *Example 2 (LOC, China):* LFS 0.9948
_"But China saw their luck desert them in the second match of the group, crashing to a surprise 2-0 defeat to newcomers Uzbekistan."_
- *Example 30 (MISC, ENGLISH):* LFS 0.9860
_"SOCCER - ENGLISH F.A. CUP SECOND ROUND RESULT."_

**Bottom 5 Examples by LFS:**

- *Example 13 (LOC, Japan):* LFS 0.2765
_"Japan coach Shu Kamo said: 'The Syrian own goal proved lucky for us."_
- *Example 31 (ORG, FIFA):* LFS 0.3054
_"Dutch forward Reggie Blinker had his indefinite suspension lifted by FIFA on Friday and was set to make his Sheffield Wednesday comeback against Liverpool on Saturday."_

---

### Visualizations

- **LFS Boxplot by Entity Type: ![image](/results/lfs_boxplot_by_entity.png)**
- **LFS Distribution: ![image](/results/lfs_distribution.png)**
- **LFS vs Sentence Length: ![image](/results/lfs_vs_sentence_length.png)**

**Additional Visualizations:**

- **LIAP-pruned attention vs. original attention: ![image](/results/visualizations/example_0.png)**
- LIME explanations for selected examples.
    - _See `results/lime/` for examples_

_See the `results/visualizations/` directory for more examples and side-by-side comparisons._

---

## How to Run

**Requirements:**

- Python 3.8+
- PyTorch, Hugging Face Transformers, spaCy, AllenNLP, LIME, matplotlib, seaborn, numpy, pandas

**Setup:**

```bash
pip install torch transformers spacy allennlp allennlp-models lime matplotlib seaborn pandas
python -m spacy download en_core_web_sm
```

**Main scripts:**

- `finetune.py`: Fine-tunes BERT on CoNLL-2003 (see code for details)
- `run_liap_evaluation.py`: Runs LIAP, LIME, and LFS evaluation, generates visualizations
- `analyze_results.py`: Summarizes and analyzes results

**Tips:**

- Ensure your model weights are saved in `best_ner_model/` with `pytorch_model.bin`.
- Adjust `run_liap_evaluation.py` to change number of examples or entity types.
- Visualizations and results will be saved in the `results/` directory.

---

## Limitations \& Future Work

- **Dependency Conflicts:** AllenNLP SRL and latest Hugging Face models require careful environment management.
- **SRL Pruning:** Only basic SRL frame overlap was used; more nuanced role-based pruning could yield further insights.
- **Dataset Size:** Results are based on 40 examples; larger-scale evaluation could further validate findings.
- **Potential Extensions:**
    - Apply LIAP to other languages or tasks (e.g., QA, sentiment analysis)
    - Explore additional linguistic features

---

## References

- [CoNLL-2003 NER Dataset](https://www.clips.uantwerpen.be/conll2003/ner/)
- [spaCy Documentation](https://spacy.io/)
- [AllenNLP SRL](https://spacy.io/universe/project/allennlp)
- [LIME: Local Interpretable Model-agnostic Explanations](https://github.com/marcotcr/lime)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- Kenny Zhu's NLP Course, UTA, Spring 2025

---

**Contact:**
Kevin Farokhrouz<br>
University of Texas at Arlington<br>
kevinafarokhrouz@gmail.com<br>
https://github.com/kevinrouz
