"""Contains definitions of CGO Optimizers"""

from typing import Iterable, Callable
from torch import nn
from torch.optim import Optimizer


class CGO(Optimizer):
    """Competitive Gradient Optimization based PyTorch Optimizer introduced in
    [Competitive Gradient Optimization](https://arxiv.org/pdf/2205.14232.pdf)
    """

    def __init__(
        self,
        param_groups: dict[nn.parameter.Parameter],
        lr: float = 0.04,
        alpha: float = 2,
        eps: float = 1e-6,
    ):
        """
        Parameters:
        param_groups (`Iterable[nn.parameter.Parameter]`):
            Dictionary of parameters to optimize defining parameter groups.
            Each dict should contain a 'player' key specifying the player id of
            that group, currently only supports 2 player games with id: 1 or 2
        lr (`float`, *optional*, defaults to 0.04):
            The learning rate to use.
        alpha (`float`, *optional*, defaults to 2):
            Interaction importance term
        eps (`float`, *optional*, defaults to 1e-6):
            epsilon for numerical stability.
        """

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate {lr} < 0.0")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha {alpha} < 0.0")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon {eps} < 0.0")
        for group in param_groups:
            if "player" not in group:
                raise ValueError(f"A Parameter group does not contain 'player' key")
            if group["player"] != 1 and group["player"] != 2:
                raise ValueError("Parameter group contains player id other than 1 or 2")
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super(CGO, self).__init__(param_groups, defaults)

    def step(self, closure: Callable = None):
        """Performs a single optimization step"""
        loss = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
