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
            Iterable with dictionary of parameters to optimize.
            Each dict should contain a 'player' key specifying the player id of
            that group, currently only supports Iterable size of 2 and
            2 player games with id: 1 or 2
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
        player1 = False
        player2 = False
        for group in param_groups:
            if "player" not in group:
                raise ValueError(f"A Parameter group does not contain 'player' key")
            if group["player"] != 1 and group["player"] != 2:
                raise ValueError("Parameter group contains player id other than 1 or 2")
            if group["player"] == 1:
                player1 = True
            if group["player"] == 2:
                player2 = True
        if not (player1 and player2):
            raise ValueError("Parameters not provided for both the players")
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super(CGO, self).__init__(param_groups, defaults)

    def step(self, closure: Callable = None):
        """Performs a single optimization step"""
        loss = None
        player_params = {"player1": None, "player2": None}
        for group in self.param_groups:
            if group["player"] == 1:
                player_params["player1"] = group
            if group["player"] == 2:
                player_params["player2"] = group
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue


#                 state = self.state[p]
#
#                 # State initialization
#                 if len(state) == 0:
#                     state["step"] = 0
#                     # Exponential moving average of gradient values
#                     state["exp_avg"] = torch.zeros_like(p.data)
#                     # Exponential moving average of squared gradient values
#                     state["exp_avg_sq"] = torch.zeros_like(p.data)
#
#                 exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
#                 beta1, beta2 = group["betas"]
#
#                 state["step"] += 1
#
#                 # Decay the first and second moment running average coefficient
#                 # In-place operations to update the averages at the same time
#                 exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
#                 exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
#                 denom = exp_avg_sq.sqrt().add_(group["eps"])
#
#                 step_size = group["lr"]
#                 if group["correct_bias"]:  # No bias correction for Bert
#                     bias_correction1 = 1.0 - beta1 ** state["step"]
#                     bias_correction2 = 1.0 - beta2 ** state["step"]
#                     step_size = (
#                         step_size * math.sqrt(bias_correction2) / bias_correction1
#                     )
#
#                 p.data.addcdiv_(exp_avg, denom, value=-step_size)
#
#                 # Just adding the square of the weights to the loss function is *not*
#                 # the correct way of using L2 regularization/weight decay with Adam,
#                 # since that will interact with the m and v parameters in strange ways.
#                 #
#                 # Instead we want to decay the weights in a manner that doesn't interact
#                 # with the m/v parameters. This is equivalent to adding the square
#                 # of the weights to the loss with plain (non-momentum) SGD.
#                 # Add weight decay at the end (fixed version)
#                 if group["weight_decay"] > 0.0:
#                     p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))
#
#         return loss
