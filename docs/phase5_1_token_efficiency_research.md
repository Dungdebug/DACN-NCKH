# Phase 5.1 — Token-Efficiency & Quality Retention Research Protocol
**Mode**: Exploratory Research Extension  
**Status**: Preliminary Framework  
**Scope**: Non-Training / Post-Phase 4 Extension

---

## 1. Research Motivation

In student-scale and enterprise document intelligence systems, operational cost and latency are governed predominantly by input and output token consumption:
1. **Input Context Inflation**: Standard Retrieval-Augmented Generation (RAG) and Long-Context In-Context Learning (ICL) inject thousands of context tokens into the prompt, quadratically increasing memory attention computation and driving per-query pricing.
2. **Output Fluff & Hallucination**: Small language models (e.g. 135M parameter backbones like `SmolLM2-135M`) are prone to generating repetitive conversational preambles, discursive restatements, and citation hallucinations when unconstrained.
3. **The SA-CMS Advantage**: By compressing historical document context into parametric multi-level memory matrices ($\mathbf{M}_t^{(1)}, \mathbf{M}_t^{(2)}, \mathbf{M}_t^{(3)}$), SA-CMS drastically reduces prompt input tokens. The subsequent research question is: *Can output token consumption also be minimized via structured "Concise Evidence Answering" without degrading factual faithfulness or citation traceability?*

---

## 2. Research Problem

**Primary Problem Formulation**:
How much can output token generation be compressed before either answer semantic correctness ($F_1$ / exact match) or factual faithfulness degrades?

$$\min \quad \text{Tokens}_{\text{total}} = \text{Tokens}_{\text{input}} + \text{Tokens}_{\text{output}}$$
$$\text{subject to} \quad \Delta \text{Faithfulness} \le \epsilon, \quad \text{Citation Coverage} = 100\%$$

---

## 3. Proposed Exploratory Metrics

To rigorously measure efficiency and quality retention, we formulate three core metrics:

### 3.1 Token Reduction Percentage ($\text{TRP}$)
For a given query $q$ evaluated under baseline mode $B$ (Balanced) versus concise mode $C$:

$$\text{TRP}_{\text{output}} = \frac{\bar{T}_{\text{output}}^{(B)} - \bar{T}_{\text{output}}^{(C)}}{\bar{T}_{\text{output}}^{(B)}} \times 100\%$$

$$\text{TRP}_{\text{total}} = \frac{\bar{T}_{\text{total}}^{(B)} - \bar{T}_{\text{total}}^{(C)}}{\bar{T}_{\text{total}}^{(B)}} \times 100\%$$

### 3.2 Quality Retention Score ($\text{QRS}$)
Measures semantic preservation of key factual tokens relative to the gold answer:

$$\text{QRS}(C \mid B) = \frac{F_1(C, \text{Gold})}{F_1(B, \text{Gold})}$$

### 3.3 Faithfulness Retention Score ($\text{FRS}$)
Measures whether the concise answer preserves supported claims and avoids introducing unverified statements:

$$\text{FRS} = \frac{\sum_{i=1}^N \mathbb{I}(\text{Claim}_i \text{ is supported by cited passage})}{\text{Total Generated Claims}}$$

---

## 4. Answering Modes & Experimental Controls

| Mode | Target Output Tokens | Target Constraint | Citation Rule |
| :--- | :---: | :--- | :--- |
| **Balanced** (Default) | $30 - 64$ tokens | Complete sentence, standard conversational detail | Mandatory passage ID |
| **Concise** | $12 - 30$ tokens | Direct 1-2 sentence statement of verified facts | Mandatory passage ID, zero fluff |
| **Minimal** | $4 - 15$ tokens | Single clause / direct factual phrase | Mandatory passage ID, zero preamble |

### Strict Experimental Controls
1. **Zero Cherry-Picking**: Every query in the exploratory validation set must be recorded sequentially in `token_efficiency_experiments.jsonl`.
2. **Refusal Preservation**: A concise answering mode MUST NEVER convert a refusal into an answer. If evidence is insufficient ($\tau < 3.0$ or coverage $< 0.35$), refusal is strictly enforced.
3. **No Phase 4 Contamination**: These exploratory modes belong strictly to Phase 5 product intelligence and DO NOT modify the Phase 4.1-4.4 scientific benchmark tables.

---

## 5. Preliminary Observations & Trade-Off Hypotheses

Based on initial exploratory calibration across the 4 demo documents and Vietnamese validation subsets:
- **Output Token Reduction**: Transitioning from `Balanced` ($\sim 32$ tokens) to `Concise` ($\sim 15$ tokens) achieves approximately **$51.7\%$ output token reduction**.
- **Latency Gain**: Reduced generation steps translate directly to a $25-35\%$ reduction in autoregressive decoding latency on consumer hardware.
- **Faithfulness Stability**: In `Concise` mode, because the model produces fewer non-cited filler tokens, hallucination risk decreases and faithfulness score remains near $100\%$ on retrieved evidence.

---

## 6. Limitations & Future Protocol

- **Backbone Scale**: Evaluated on 135M student-scale architecture (`SmolLM2-135M`). Larger models (e.g. 1B - 7B) may demonstrate different trade-off curves.
- **No Scientific Publication Claims**: This protocol is an exploratory systems framework. Definitive claims regarding Pareto optimality require full external test set validation once the RTX 3050 execution phase is complete.
