import torch


def l1_loss(network_output: torch.Tensor, gt: torch.Tensor, agg="mean") -> torch.Tensor:
    """
    Computes the L1 loss, which is the mean absolute error between the network output and the ground truth.

    Args:
        network_output: The output from the network.
        gt: The ground truth tensor.
        agg: The aggregation method to be used. Defaults to "mean".
    Returns:
        The computed L1 loss.
    """
    l1_loss = torch.abs(network_output - gt)
    if agg == "mean":
        return l1_loss.mean()
    elif agg == "sum":
        return l1_loss.sum()
    elif agg == "none":
        return l1_loss
    else:
        raise ValueError("Invalid aggregation method.")


def isotropic_loss(scaling: torch.Tensor) -> torch.Tensor:
    """
    Computes loss enforcing isotropic scaling for the 3D Gaussians
    Args:
        scaling: scaling tensor of 3D Gaussians of shape (n, 3)
    Returns:
        The computed isotropic loss
    """
    mean_scaling = scaling.mean(dim=1, keepdim=True)
    isotropic_diff = torch.abs(scaling - mean_scaling * torch.ones_like(scaling))
    return isotropic_diff.mean()
