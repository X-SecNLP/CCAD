# Formal Robustness Verification

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
