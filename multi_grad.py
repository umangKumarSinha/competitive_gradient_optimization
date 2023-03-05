import torch
from torch import autograd


def mixed_derivative(f, x, y):
    hessian = []
    # calculate grad wrt one set of parameters to create a vector
    first_derivative = autograd.grad(f, x, create_graph=True, allow_unused=True)[0]
    print(f"first derivative: {first_derivative}")
    # iteratively calculate grad wrt other set of parameters to create 2D tensor
    for param in first_derivative:
        second_derivative = autograd.grad(
            param, y, allow_unused=True, retain_graph=True
        )[0]
        print(f"second derivative: {second_derivative}")
        hessian.append(second_derivative)
    return hessian


x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.tensor([2.0, 1.0], requires_grad=True)
f = torch.matmul(x, y)
print(mixed_derivative(f, x, y))
