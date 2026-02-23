import torch
import torch.nn as nn
from z3 import *

class CCADFormalVerifier:
    def __init__(self, model_head, threshold=0.5):
        """
        :param model_head: A torch.nn.Sequential model (Linear + ReLU layers)
        :param threshold: Anomaly detection threshold (Score > threshold is Anomaly)
        """
        self.model = model_head
        self.threshold = float(threshold)
        self.layers = [m for m in model_head.modules() if isinstance(m, nn.Linear)]

    def verify(self, input_data, epsilon=0.01):
        """
        Verifies if the model output remains stable within the epsilon-neighbor of input_data.
        :param input_data: Original feature vector (torch.Tensor)
        :param epsilon: L-infinity perturbation bound
        """
        solver = Solver()
        
        # 1. Define input symbolic variables
        input_np = input_data.detach().cpu().numpy().flatten()
        n_in = len(input_np)
        x_vars = [Real(f'x_0_{i}') for i in range(n_in)]
        
        # 2. Add input constraints (Input space box constraints)
        for i in range(n_in):
            solver.add(x_vars[i] >= float(input_np[i] - epsilon))
            solver.add(x_vars[i] <= float(input_np[i] + epsilon))

        # 3. Encode Network Layers into Z3 Constraints
        current_vars = x_vars
        for layer_idx, layer in enumerate(self.layers):
            weight = layer.weight.detach().cpu().numpy()
            bias = layer.bias.detach().cpu().numpy()
            n_out = layer.out_features
            next_vars = [Real(f'x_{layer_idx+1}_{i}') for i in range(n_out)]
            
            for i in range(n_out):
                # Compute Linear Transformation: y = Wx + b
                linear_expr = Sum([float(weight[i][j]) * current_vars[j] for j in range(len(current_vars))]) + float(bias[i])
                
                # Apply ReLU if it is not the last layer
                if layer_idx < len(self.layers) - 1:
                    solver.add(next_vars[i] == If(linear_expr > 0, linear_expr, 0))
                else:
                    # Final output score (Raw Logit)
                    solver.add(next_vars[i] == linear_expr)
            
            current_vars = next_vars

        # 4. Define Adversarial Goal
        # Goal: Find an input x' such that original score was < threshold, but x' score >= threshold
        final_score = current_vars[0]
        solver.add(final_score >= self.threshold)

        # 5. Execute Solver
        result = solver.check()
        
        if result == sat:
            print(f"Verification Failed: Model is unstable at epsilon {epsilon}")
            m = solver.model()
            # Extract the adversarial input found by Z3
            adv_input = [float(m[x_vars[i]].as_decimal(6).replace('?', '')) for i in range(n_in)]
            return False, adv_input
        else:
            print(f"Verification Success: Model is formally robust at epsilon {epsilon}")
            return True, None

# Example Usage for CCAD
if __name__ == "__main__":
    # Define a simple projection head: 8 -> 16 (ReLU) -> 1
    ccad_head = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    
    # Simulate a 'Normal' sample feature vector
    sample_feature = torch.rand(1, 8)
    
    verifier = CCADFormalVerifier(ccad_head, threshold=0.7)
    
    # Check robustness within a 0.05 perturbation range
    is_robust, counter_example = verifier.verify(sample_feature, epsilon=0.05)
    
    if counter_example:
        print("Adversarial Feature Vector:", counter_example)
