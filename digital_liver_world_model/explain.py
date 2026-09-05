"""
Integrated Gradients for explaining world-model predictions.

Targets a specific feature at a specific timestep so the attribution
answers questions like "why did the model predict decompensation (P) at month 30?"
"""

import torch


def explain_sample(model, x, context=None, baseline=None, steps=50,
                   target_timestep=-1, target_feature=3):
    if baseline is None:
        baseline = x.mean(dim=1, keepdim=True).expand_as(x)
    x = x.clone().detach()
    alphas = torch.linspace(0, 1, steps, device=x.device)
    grad_sum = torch.zeros_like(x)
    for i, alpha in enumerate(alphas):
        x_scaled = baseline + alpha * (x - baseline)
        x_scaled = x_scaled.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            z = model.encode(x_scaled, context=context)
            z_pred = model.predict_latent(z)
            x_pred = model.decode(z_pred)
            pred = x_pred[:, -model.forecast_horizon:]
            loss = pred[0, target_timestep, target_feature]
        grad = torch.autograd.grad(loss, x_scaled, create_graph=False)[0]
        if i == 0 or i == steps - 1:
            grad_sum += 0.5 * grad.detach()
        else:
            grad_sum += grad.detach()
    ig = (x - baseline) * grad_sum / (steps - 1)
    return ig.squeeze(0)
