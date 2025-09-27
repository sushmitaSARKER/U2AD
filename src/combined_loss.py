import torch
import torch.nn as nn

from .sampling import get_prediction
from .contextual_information_gain import contextual_information_gain


def compute_loss_components(
        x,
        sde_model,
        score_model,
        eps,
        device,
        win_size,
        likelihood_weighting=False,
        is_rec_loss=True,
        is_sde_loss=True,
        is_attention_loss=True,
        is_svdd_loss=True,
        svdd_criterion=None,
        ode_solver=True,
):
    """
    Compute the loss function components for training score-based generative models.

    Args:
        x: A mini-batch of training data.
        sde_model: The SDE model instance.
        score_model: A PyTorch model instance that represents a time-dependent score-based model.
        eps: A tolerance value for numerical stability.
        device: Device to run computations on.
        win_size: Window size for sequences.
        likelihood_weighting: Whether to use likelihood weighting.
        is_rec_loss: Whether to compute reconstruction loss.
        is_sde_loss: Whether to compute SDE loss.
        is_attention_loss: Whether to compute contextual information gain loss.
        is_svdd_loss: Whether to compute SVDD loss.
        svdd_criterion: SVDD loss criterion.
        ode_solver: Whether to use ODE solver for reconstruction.

    Returns:
        tuple: (loss_sde, loss_rec, loss_global_context, loss_local_context, loss_svdd)
    """
    # Sample random time steps
    t = torch.rand(x.shape[0], device=x.device) * (sde_model.T - eps) + eps
    z = torch.randn_like(x)
    mean, std = sde_model.marginal_prob(x, t)
    perturbed_x = (mean + std[:, None, None] * z).to(device)
    
    # Score generation with score model
    score, global_context, local_context, _ = score_model(perturbed_x, t)

    # SDE Loss calculation
    if is_sde_loss:
        if not likelihood_weighting:
            loss_sde = torch.square(score * std[:, None, None] + z)
            loss_sde = torch.mean(loss_sde.reshape(loss_sde.shape[0], -1), dim=-1)
        else:
            g2 = sde_model.sde(torch.zeros_like(x), t)[1] ** 2
            loss_sde = torch.square(score + z / std[:, None, None])
            loss_sde = torch.mean(loss_sde.reshape(loss_sde.shape[0], -1), dim=-1) * g2
    else:
        loss_sde = 0

    # Reconstruction Loss calculation
    if is_rec_loss:
        x_hat = get_prediction(
            perturbed_x, 
            t, 
            score,
            device=device,
            sde_model=sde_model,
            sde_solver='ode' if ode_solver else 'euler_maruyama',
            eps=eps
        )
        loss_rec = nn.MSELoss()(x_hat, x)
    else:
        loss_rec = 0

    # Contextual information gain calculation
    if is_attention_loss:
        loss_global_context, loss_local_context = contextual_information_gain(
            global_context,
            local_context,
            win_size=win_size
        )
    else:
        loss_global_context, loss_local_context = 0, 0

    # SVDD loss calculation
    if is_svdd_loss and svdd_criterion is not None:
        loss_svdd = svdd_criterion(score)
    else:
        loss_svdd = 0

    return loss_sde, loss_rec, loss_global_context, loss_local_context, loss_svdd


def compute_final_loss(
        loss_sde,
        loss_rec,
        loss_global_context,
        loss_local_context,
        loss_svdd,
        lamb_k=1.0,
        k_1=1.0,
        k_2=1.0,
        normalizing_factor=100,
        lamb_k_2=1.0
):
    """
    Compute the final combined loss from individual loss components.

    Args:
        loss_sde: SDE loss component.
        loss_rec: Reconstruction loss component.
        loss_global_context: Global context loss component.
        loss_local_context: Local context loss component.
        loss_svdd: SVDD loss component.
        lamb_k: Weight for global_context attention discrepancy.
        k_1: Weight for global_context in attention discrepancy.
        k_2: Weight for local_context in attention discrepancy.
        normalizing_factor: Factor to normalize reconstruction loss.
        lamb_k_2: Weight for SVDD loss.

    Returns:
        tuple: (loss_combined, global_context_adj_loss, local_context_adj_loss)
    """
    # Normalize reconstruction loss
    loss_rec_normalized = loss_rec / normalizing_factor
    
    # Compute adjusted attention losses
    loss_global_context_adj = loss_rec_normalized - k_1 * loss_global_context
    loss_local_context_adj = loss_rec_normalized + k_2 * loss_local_context
    
    # Compute final combined loss
    loss_combined = loss_sde + lamb_k * loss_global_context_adj + lamb_k_2 * loss_svdd

    return loss_combined, loss_global_context_adj, loss_local_context_adj