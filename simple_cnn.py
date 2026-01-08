import numpy as np


def relu(Z: np.ndarray) -> np.ndarray:
    return np.maximum(Z, 0.0)


def relu_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return dA * (Z > 0)


def softmax(Z: np.ndarray) -> np.ndarray:
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def one_hot(y: np.ndarray, K: int) -> np.ndarray:
    Y = np.zeros((y.shape[0], K), dtype=np.float32)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def cross_entropy(P: np.ndarray, Y: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(np.sum(Y * np.log(P + eps), axis=1)))


def accuracy_from_logits(Z: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(Z, axis=1)
    return float(np.mean(pred == y))


def conv2d_forward_nchw(X, W, b, stride=1, pad=1):
    N, C, H, W_in = X.shape
    F, Cw, HH, WW = W.shape
    assert C == Cw
    assert b.shape == (F,)

    H_out = (H + 2 * pad - HH) // stride + 1
    W_out = (W_in + 2 * pad - WW) // stride + 1

    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    Y = np.zeros((N, F, H_out, W_out), dtype=np.float32)
    cache = (X, W, b, stride, pad, Xp)
    return Y, cache


if __name__ == "__main__":
    x = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
    print("relu:", relu(x))
    dA = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    print("relu_backward:", relu_backward(dA, x))
    z = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    print("softmax:", softmax(z))
    y = np.array([0, 2], dtype=np.int64)
    print("one_hot:", one_hot(y, 3))
    y2 = np.array([2], dtype=np.int64)
    P = softmax(z)
    Y = one_hot(y2, 3)
    print("cross_entropy:", cross_entropy(P, Y))
    z2 = np.array([[0.1, 0.9, 0.0], [0.2, 0.1, 0.7]], dtype=np.float32)
    y3 = np.array([1, 2], dtype=np.int64)
    print("accuracy:", accuracy_from_logits(z2, y3))
