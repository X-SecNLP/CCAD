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

----

## Echoes of Tomorrow

![img](https://github.com/user-attachments/assets/6cefdb8d-1b81-4ebb-82c0-745bd3ec2d4c)
