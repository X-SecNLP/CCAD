import torch
from models import BertMLMSynonymReplacer, SemanticsModel
from utils import clip_duel_loss, edit_distance
from typing import List

class CCAD_Attack:
    def __init__(self, model, processor, epsilon, step_size, num_iterations, alpha, beta, gamma, device):
        self.model = model
        self.processor = processor
        self.epsilon, self.step_size = epsilon, step_size
        self.num_iterations = num_iterations
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.device = device
        self.synonym_replacer = BertMLMSynonymReplacer() 
        self.semantics_model = SemanticsModel(device=device)

    def evaluate(self, I_adv, T_adv_text, T_orig_text, mode):
        I_orig = self.initial_I_orig.data
        delta_I = I_adv.data - I_orig
        text_inputs = self.processor(text=T_adv_text, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'], pixel_values=I_adv.detach())
        
        target_T_orig = self.initial_T_orig_text if mode != "I" else T_adv_text
        return clip_duel_loss(outputs.image_embeds, outputs.text_embeds, delta_I, T_adv_text, target_T_orig, self.alpha, self.beta, self.gamma, self.semantics_model)

    def _attack_image_pgd(self, I_orig, delta_I, T_adv_text, mode):
        delta_I.requires_grad = True
        I_adv = torch.clamp(I_orig + delta_I, 0, 1)
        text_inputs = self.processor(text=T_adv_text, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'], pixel_values=I_adv)
        
        T_orig_p = T_adv_text if mode == "I" else [self.initial_T_orig_text[0]]
        total_loss, _, _, _ = clip_duel_loss(outputs.image_embeds, outputs.text_embeds, delta_I, T_adv_text, T_orig_p, self.alpha, self.beta, self.gamma, self.semantics_model)
        
        self.model.zero_grad()
        total_loss.backward()
        
        with torch.no_grad():
            grad_sign = delta_I.grad.data.sign()
            new_delta = delta_I.data - self.step_size * grad_sign
            new_delta = torch.clamp(new_delta, -self.epsilon, self.epsilon)
            new_delta = torch.clamp(I_orig + new_delta, 0, 1) - I_orig
            
        return new_delta.detach(), total_loss.item()

    def _attack_text_synonym(self, I_adv, T_adv_text, T_orig_text, mode):
        T_adv_list = T_adv_text[0].split()
        best_T = T_adv_text[0]
        current_loss = self.evaluate(I_adv, T_adv_text, T_orig_text, mode)[0].item()
        max_reduction = 0
        
        for i, word in enumerate(T_adv_list):
            if len(word) < 3 or not word.isalpha(): continue 
            candidates = self.synonym_replacer.get_candidates(T_adv_text[0], i)
            for cand in candidates:
                T_new_list = T_adv_list[:]; T_new_list[i] = cand
                T_new_text = [" ".join(T_new_list)]
                new_loss = self.evaluate(I_adv, T_new_text, T_orig_text, mode)[0].item()
                if current_loss - new_loss > max_reduction:
                    max_reduction = current_loss - new_loss
                    best_T = T_new_text[0]
        return [best_T], current_loss - max_reduction

    def run_attack(self, I_orig: torch.Tensor, T_orig_text: List[str], attack_mode: str):
        self.initial_I_orig = I_orig.clone().detach().to(self.device)
        self.initial_T_orig_text = T_orig_text[:]
        
        if attack_mode != "T":
            delta_I = torch.empty_like(self.initial_I_orig).uniform_(-self.epsilon, self.epsilon).to(self.device)
            delta_I.data = torch.clamp(self.initial_I_orig + delta_I.data, 0, 1) - self.initial_I_orig
        else:
            delta_I = torch.zeros_like(self.initial_I_orig).to(self.device)
            
        T_adv_text = [T_orig_text[0]]
        init_loss, init_sim, init_ri, init_rt = self.evaluate(self.initial_I_orig, T_adv_text, self.initial_T_orig_text, attack_mode)
        initial_metrics = (init_loss.item(), init_sim.item(), init_ri.item(), init_rt)

        best_loss = float('inf')
        best_data = {'image': None, 'text': None, 'metrics': None, 'iter': -1}
        history = {'loss': [], 'sim_loss': [], 'R_I': [], 'R_T': [], 'text': []}

        print(f"\n--- Starting {attack_mode} CCAD Attack ---")

        for i in range(self.num_iterations):
            if attack_mode == "I":
                delta_I, _ = self._attack_image_pgd(self.initial_I_orig, delta_I, T_adv_text, "I")
            elif attack_mode == "T":
                T_adv_text, _ = self._attack_text_synonym(self.initial_I_orig, T_adv_text, self.initial_T_orig_text, "T")
            elif attack_mode == "C":
                delta_I, _ = self._attack_image_pgd(self.initial_I_orig, delta_I, T_adv_text, "C")
                I_adv_upd = torch.clamp(self.initial_I_orig + delta_I.data, 0, 1).detach()
                T_adv_text, _ = self._attack_text_synonym(I_adv_upd, T_adv_text, self.initial_T_orig_text, "C")

            I_adv_final = torch.clamp(self.initial_I_orig + delta_I.data, 0, 1)
            loss_t, sim_t, ri_t, rt_t = self.evaluate(I_adv_final, T_adv_text, self.initial_T_orig_text, attack_mode)
            loss_val = loss_t.item()

            if loss_val < best_loss:
                best_loss = loss_val
                best_data.update({
                    'image': I_adv_final.clone().detach(),
                    'text': T_adv_text[0],
                    'metrics': (loss_val, sim_t.item(), ri_t.item(), rt_t),
                    'iter': i + 1
                })

            if (i+1) % 5 == 0 or i == self.num_iterations - 1:
                print(f"Iter {i+1}/{self.num_iterations} | Loss: {loss_val:.4f} | Best: {best_loss:.4f}")

        return I_adv_final.data, T_adv_text[0], history, best_data, initial_metrics
