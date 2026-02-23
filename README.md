# CCAD: Cooperative Cross-modal Adversarial Duel

This repository implements **CCAD**, a modular framework for performing adversarial attacks on CLIP (Contrastive Language-Image Pre-training) models.

It employs a dual-strategy approach: **Projected Gradient Descent (PGD)** for image perturbations and **BERT-MLM** for context-aware synonym replacement in text.

## Project Structure

| File | Description |
| --- | --- |
| `models.py` | Contains `BertMLMSynonymReplacer` and `SemanticsModel` for text processing. |
| `utils.py` | Core loss functions (`clip_duel_loss`) and distance metrics. |
| `attacker.py` | The main `CCAD_Attack` engine implementing PGD and collaborative steps. |
| `main.py` | Entry point for configuration, data loading, and execution. |

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/X-SecNLP/CCAD.git
cd ./ccad

```

2. **Install dependencies**:

## Usage

### Quick Start

1. Place an image (e.g., `cat.jpg`) in the project root.
2. Run the attack script:
```bash
python main.py

```

### Configuration

You can adjust the attack intensity in `main.py` via the `params` dictionary:

* `epsilon`: The maximum perturbation allowed for pixels (e.g., `2/255.0`).
* `alpha`/`beta`: Weighting factors for image and text penalties.
* `gamma`: Balance between semantic similarity and edit distance for text.

## Methodology

### The CLIP Duel Loss

The framework optimizes the following objective function:

Where:

* Cosine similarity between CLIP embeddings.
* A weighted penalty of semantic shift (via MiniLM) and Word Edit Distance.

### Attack Modes

* **Mode "I"**: Updates only the image using PGD gradients.
* **Mode "T"**: Updates only the text using BERT-based greedy search.
* **Mode "C"**: Alternates between "I" and "T" steps in each iteration for a collaborative effect.

## Expected Output

Upon execution, the script generates a comparison table showing the attack's effectiveness:

```text
===========================================================================
                          CCAD PGD Attack Result                           
===========================================================================
Metric                 | Initial              | Best                
---------------------------------------------------------------------------
Total Loss             | 0.2765               | 0.5688              
Cosine Sim Loss        | 0.2765               | 0.1908              
---------------------------------------------------------------------------
Initial Text: A photo of a cat staring forward.
Best Text:    A silhouette of a cat staring forward.
===========================================================================
```

## Requirements

* Google Colab - T4 GPU

# Formal Robustness Verification (SMT-Powered)

This implementation of **CCAD** integrates an **SMT-based Formal Verification** module using the **Z3 Solver**. Unlike empirical testing (e.g., PGD or FGSM attacks) which only "try" to find errors, this module provides a mathematical guarantee of the model's decision stability.

## Overview
In safety-critical anomaly detection, it is vital to know if a model's "Normal" classification can be flipped by minor input noise. We translate the neural network's **Linear** and **ReLU** operations into first-order logic constraints:

1. **Input Constraints**: Define an $L_\infty$ ball around the input $x$ with radius $\epsilon$.
2. **Layer Constraints**: Model each neuron as a symbolic variable where $y = \max(0, Wx + b)$.
3. **Safety Property**: Search for any $x'$ within the $\epsilon$-ball such that $f(x') \geq \text{threshold}$ (False Anomaly).

## Core Features
* **Deterministic Bounds**: Mathematically prove that no perturbation exists within the defined range that can deceive the model.
* **Counter-example Generation**: If the model is vulnerable, Z3 returns the exact feature vector that triggers the failure.
* **Latent Space Analysis**: Optimized for the CCAD projection head to ensure efficient solving times.

## Usage
To verify the robustness of your trained CCAD model scoring head:

```python
from z3_verifier import CCADFormalVerifier

# 1. Initialize verifier with the trained scoring head
verifier = CCADFormalVerifier(model.projection_head, threshold=0.8)

# 2. Run verification on a latent feature vector
# Returns True if formally robust, False if a vulnerability is found
is_robust, counter_example = verifier.verify(feature_tensor, epsilon=0.05)

if is_robust:
    print("Decision is formally guaranteed within epsilon 0.05.")
else:
    print(f"Vulnerability found! Adversarial feature: {counter_example}")
```
