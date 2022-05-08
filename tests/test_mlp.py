import numpy as np
import pytest

from mlp_numpy import MLP


def mse_for_test_backward(target):
    """MSE for use in ``test_backward``."""
    grad_loss = lambda forecast: forecast - target
    loss = lambda forecast: (0.5 * (forecast - target)**2).sum(axis=1)  # Multiple heads are weighted equally.
    return grad_loss, loss


@pytest.mark.parametrize("layer_sizes,n_samples,setup_loss,transform", [
    ((32, 16, 8, 1), 100, mse_for_test_backward, lambda x: x),
    ((32, 16, 8, 2), 100, mse_for_test_backward, np.sin),
])
def test_backward(layer_sizes, n_samples, setup_loss, transform):
    """Validate gradients computed by backward against those computed by finite differences."""
    # pylint: disable=too-many-locals
    mlp = MLP(layer_sizes=layer_sizes)
    batch = np.random.randn(n_samples, mlp.n_features)
    target = np.random.randn(n_samples, mlp.n_heads)
    grad_loss, loss = setup_loss(target)
    grads = mlp.backward(*mlp.forward(batch), grad_loss=grad_loss, transform=transform)
    n_hidden = len(mlp.layers) - 2
    for layer_idx in range(1, (n_hidden + 1) + 1):
        grad_weight, grad_bias = grads[layer_idx]
        for i in range(grad_weight.shape[0]):
            for j in range(grad_weight.shape[1]):
                grad = transform(mlp.fd_grad_weight(batch, (i, j), layer_idx, loss)).mean()
                assert np.isclose(grad_weight[i, j], grad, rtol=1e-3)
        for i in range(grad_bias.shape[0]):
            grad = transform(mlp.fd_grad_bias(batch, i, layer_idx, loss)).mean()
            assert np.isclose(grad_bias[i], grad, rtol=1e-3)
