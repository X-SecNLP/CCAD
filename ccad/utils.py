import torch

def edit_distance(text1: str, text2: str) -> int:
    words1, words2 = text1.lower().split(), text2.lower().split()
    if text1 == text2: return 0
    diff_count = 0
    for i in range(max(len(words1), len(words2))):
        w1 = words1[i] if i < len(words1) else ""
        w2 = words2[i] if i < len(words2) else ""
        if w1 != w2: diff_count += 1
    return diff_count

def clip_duel_loss(image_features, text_features, delta_I, T_adv_text, T_orig_text, alpha, beta, gamma, semantics_model):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    sim_loss = torch.sum(image_features * text_features) / image_features.size(0)
    
    R_I = torch.norm(delta_I, p=float('inf'))
    
    if T_adv_text[0] == T_orig_text[0]:
        R_T = 0.0
    else:
        sem_sim = semantics_model.semantic_similarity(T_adv_text[0], T_orig_text[0])
        edit_count = edit_distance(T_adv_text[0], T_orig_text[0])
        R_T = gamma * (1.0 - sem_sim) + (1.0 - gamma) * edit_count
        
    total_loss = sim_loss + alpha * R_I + beta * R_T
    return total_loss, sim_loss, R_I, R_T
