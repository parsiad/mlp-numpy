from typing import Callable, List, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound=npt.NBitBase)
FloatArray = npt.NDArray[np.floating[T]]
FloatArrayPairs = List[Tuple[FloatArray, FloatArray]]


# TODO(parsiad): Support arbitrary activation functions.
class MLP:
    """A multi-layer perceptron for regression. All intermediate activations are ReLU.

    Parameters
    ----------
    layer_sizes: Sequence of layer sizes (e.g., (32, 16, 8, 1)).
    rng: Random number generator.
    """
    def __init__(
        self,
        layer_sizes: Tuple[int, ...],
        rng: Optional[np.random.Generator] = None,
    ):
        if len(layer_sizes) < 2:
            raise ValueError('At the minimum, an input and output layer must be present')
        rng_: np.random.Generator = np.random.default_rng(0) if rng is None else rng
        self._layer_sizes = layer_sizes
        self._layers: FloatArrayPairs = [(np.array(np.NaN), np.array(np.NaN))]
        n_hidden = len(self._layer_sizes) - 2
        for layer_idx in range(1, (n_hidden + 1) + 1):
            weight = rng_.standard_normal(size=(self._layer_sizes[layer_idx - 1], self._layer_sizes[layer_idx]))
            bias = np.zeros((self._layer_sizes[layer_idx], ))
            weight /= np.sqrt(self._layer_sizes[layer_idx - 1])  # Xavier init
            self._layers.append((weight, bias))

    @property
    def layers(self) -> FloatArrayPairs:
        """Returns a copy of the weights and biases."""
        return [(weight.copy(), bias.copy()) for weight, bias in self._layers]

    @property
    def n_features(self) -> int:
        """"Returns the number of features used by this network."""
        return self._layer_sizes[0]

    @property
    def n_heads(self) -> int:
        """Returns the number of heads forecast by this network."""
        return self._layer_sizes[-1]

    def forward(
        self,
        batch: FloatArray,
    ) -> Tuple[FloatArray, FloatArrayPairs]:
        """Computes the forward pass.

        Parameters
        ----------
        batch: ``(n_samples, n_features)`` shaped input batch.

        Returns
        -------
        forecast: ``(n_samples, n_heads)`` shaped forecast.
        aux: Auxiliary information needed for the backward pass.
        """
        return MLP._forward(batch, self._layers)

    @staticmethod
    def _forward(
        batch: FloatArray,
        layers: FloatArrayPairs,
    ) -> Tuple[FloatArray, FloatArrayPairs]:
        if batch.ndim != 2:
            raise ValueError('Batch expected to have two dimensions')
        aux: FloatArrayPairs = [(np.array(np.NaN), batch)]
        neuron = batch
        n_hidden = len(layers) - 2
        layer_idx = 1
        while True:
            weight, bias = layers[layer_idx]
            affine = np.dot(neuron, weight) + bias
            if layer_idx == n_hidden + 1:
                break
            neuron = np.maximum(affine, 0.)
            aux.append((affine, neuron))
            layer_idx += 1
        forecast = affine
        return forecast, aux

    def backward(
        self,
        forecast: FloatArray,
        aux: FloatArrayPairs,
        grad_loss: Callable[[FloatArray], FloatArray],
        transform: Callable[[FloatArray], FloatArray] = lambda x: x,
    ) -> FloatArrayPairs:
        """Computes the backward pass.

        Parameters
        ----------
        forecast: ``(n_samples, n_heads)`` shaped forecast from the forward pass.
        aux: Auxiliary information from the forward pass.
        grad_loss: Gradient of the loss. For example, MSE is obtained by using ``lambda forecast: forecast - target``.
            The output of this function should have shape ``(n_samples, n_heads)``.
        transform: Element-wise mapping from a floating point array to a floating point array. If ``transform`` is not
            specified, the expected value of the gradient of a particular weight ``w`` is approximated using samples
            from the batch. This quantity is exactly ``E[grad_w(loss(Y_hat, Y))]``. The ``transform`` argument enables
            the more general expression ``E[transform(grad_w(loss(Y_hat, Y)))]``.

        Returns
        -------
        grad_pairs: List in which each entry corresponds to the gradient of a layer. Each element is a 2-tuple
            containing the gradient of the weights and the gradient of the biases, in that order.
        """
        n_hidden = len(self._layers) - 2
        layer_idx = n_hidden + 1
        chain : FloatArray = np.expand_dims(grad_loss(forecast), axis=1)
        grads = []
        while True:
            affine, neuron = aux[layer_idx - 1]
            grad_weight = transform(chain * np.expand_dims(neuron, axis=-1)).mean(axis=0)
            grad_bias = transform(chain).mean(axis=0).squeeze(axis=0)
            grads.append((grad_weight, grad_bias))
            if layer_idx == 1:
                break
            weight, _ = self._layers[layer_idx]
            chain = chain @ (weight.T * np.expand_dims(affine > 0, axis=1))
            layer_idx -= 1
        grads.append((np.array(np.NaN), np.array(np.NaN)))
        return grads[::-1]

    def fd_grad_weight(
        self,
        batch: FloatArray,
        idx: Tuple[int, int],
        layer_idx: int,
        loss: Callable[[FloatArray], FloatArray],
    ) -> FloatArray:
        """Computes weight gradients by finite differences (this is slow and should only be used for testing)."""
        offset = 1e-6
        layers_offset = self.layers
        layers_offset[layer_idx][0][idx[0], idx[1]] += offset
        return self._fd_grad(batch, layers_offset, loss, offset)

    def fd_grad_bias(
        self,
        batch: FloatArray,
        idx: int,
        layer_idx: int,
        loss: Callable[[FloatArray], FloatArray],
    ) -> FloatArray:
        """Computes bias gradients by finite differences (this is slow and should only be used for testing)."""
        offset = 1e-6
        layers_offset = self.layers
        layers_offset[layer_idx][1][idx] += offset
        return self._fd_grad(batch, layers_offset, loss, offset)

    def _fd_grad(
        self,
        batch: FloatArray,
        layers_offset: FloatArrayPairs,
        loss: Callable[[FloatArray], FloatArray],
        offset: float,
    ) -> FloatArray:
        forecast_offset, _ = MLP._forward(batch, layers_offset)
        forecast, _ = MLP._forward(batch, self._layers)
        grad = (loss(forecast_offset) - loss(forecast)) / offset
        return grad
