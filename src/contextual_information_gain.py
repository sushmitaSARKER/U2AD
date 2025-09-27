import torch
from .utils import my_kl_loss

def contextual_information_gain(gloabl_context, local_context, win_size):
    """
    Compute contextual information gain between gloabl_context and local_context.
    
    Returns:
        tuple: (gloabl_context_loss, local_context_loss)
    """
    gloabl_context_loss = 0.0
    local_context_loss = 0.0
    
    for u in range(len(local_context)):
        # Normalize local_context distribution
        local_context_sum = torch.unsqueeze(torch.sum(local_context[u], dim=-1), dim=-1)
        normalized_local_context = local_context[u] / local_context_sum.repeat(1, 1, 1, win_size)
        normalized_local_context = normalized_local_context.detach()
        
        # Compute bidirectional KL divergence for gloabl_context
        gloabl_context_kl_1 = my_kl_loss(gloabl_context[u], normalized_local_context)
        gloabl_context_kl_2 = my_kl_loss(normalized_local_context, gloabl_context[u])
        gloabl_context_loss += torch.mean(gloabl_context_kl_1) + torch.mean(gloabl_context_kl_2)
        
        # Compute bidirectional KL divergence for local_context  
        local_context_kl_1 = my_kl_loss(normalized_local_context, gloabl_context[u].detach())
        local_context_kl_2 = my_kl_loss(gloabl_context[u].detach(), normalized_local_context)
        local_context_loss += torch.mean(local_context_kl_1) + torch.mean(local_context_kl_2)
    
    # Average over all layers
    gloabl_context_loss = gloabl_context_loss / len(local_context)
    local_context_loss = local_context_loss / len(local_context)
    
    return gloabl_context_loss, local_context_loss