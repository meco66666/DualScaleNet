import torch
import torch.nn.functional as F

def custom_similarity_loss(zx_vars, zy_vars):
    similar_pairs = [(zx_vars[i], zy_vars[i]) for i in range(9)]
    dissimilar_pairs = [(zx_vars[i], zy_vars[8 - i]) for i in range(9) if i != 4]
    
    positive_loss = 0
    for zx, zy in similar_pairs:
        cos_sim = F.cosine_similarity(zx, zy, dim=-1)
        positive_loss += 1 - cos_sim.mean()

    negative_loss = 0
    for zx, zy in dissimilar_pairs:
        cos_sim = F.cosine_similarity(zx, zy, dim=-1)
        negative_loss += cos_sim.mean()

    total_loss = 0.8*positive_loss + 0.2*negative_loss
    return total_loss
