import numpy as np


def relu(Z: np.ndarray) -> np.ndarray:
    return np.maximum(Z, 0.0)


def relu_backward(dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return dA * (Z > 0)


if __name__ == "__main__":
    x = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
    print("relu:", relu(x))
    dA = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    print("relu_backward:", relu_backward(dA, x))
