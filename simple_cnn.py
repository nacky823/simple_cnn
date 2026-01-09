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

    for n in range(N):
        for f in range(F):
            for oh in range(H_out):
                hs = oh * stride
                for ow in range(W_out):
                    ws = ow * stride
                    patch = Xp[n, :, hs:hs + HH, ws:ws + WW]
                    Y[n, f, oh, ow] = np.sum(patch * W[f]) + b[f]

    cache = (X, W, b, stride, pad, Xp)
    return Y, cache


def conv2d_backward_nchw(dY, cache):
    X, W, b, stride, pad, Xp = cache
    N, C, H, W_in = X.shape
    F, _, HH, WW = W.shape
    _, _, H_out, W_out = dY.shape

    dXp = np.zeros_like(Xp, dtype=np.float32)
    dW = np.zeros_like(W, dtype=np.float32)
    db = np.zeros((F,), dtype=np.float32)
    for f in range(F):
        db[f] = np.sum(dY[:, f, :, :])

    for n in range(N):
        for f in range(F):
            for oh in range(H_out):
                hs = oh * stride
                for ow in range(W_out):
                    ws = ow * stride
                    grad = dY[n, f, oh, ow]
                    patch = Xp[n, :, hs:hs + HH, ws:ws + WW]
                    dW[f] += patch * grad
                    dXp[n, :, hs:hs + HH, ws:ws + WW] += W[f] * grad

    if pad == 0:
        dX = dXp
    else:
        dX = dXp[:, :, pad:pad + H, pad:pad + W_in]
    return dX, dW, db


def load_mnist_nchw():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = (x_train.astype(np.float32) / 255.0)
    x_test = (x_test.astype(np.float32) / 255.0)

    X_train = x_train[:, None, :, :]
    X_test = x_test[:, None, :, :]
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    return X_train, y_train, X_test, y_test


def verify_conv_forward_with_tf(X_nchw, W_fchw, b_f, stride, pad, atol=1e-4):
    import tensorflow as tf

    Y_np, _ = conv2d_forward_nchw(X_nchw, W_fchw, b_f, stride=stride, pad=pad)

    X_nhwc = np.transpose(X_nchw, (0, 2, 3, 1))
    W_hwcf = np.transpose(W_fchw, (2, 3, 1, 0))

    X_tf = tf.convert_to_tensor(X_nhwc, dtype=tf.float32)
    W_tf = tf.convert_to_tensor(W_hwcf, dtype=tf.float32)
    b_tf = tf.convert_to_tensor(b_f, dtype=tf.float32)
    if pad == 0:
        padding = "VALID"
    else:
        padding = "SAME"
        if not (pad == 1 and W_fchw.shape[2] == 3 and W_fchw.shape[3] == 3 and stride == 1):
            raise ValueError("pad=1, 3x3, stride=1 のSAMEのみ対応")

    Y_tf = tf.nn.conv2d(X_tf, W_tf, strides=[1, stride, stride, 1], padding=padding)
    Y_tf = (Y_tf + b_tf).numpy()
    Y_tf_nchw = np.transpose(Y_tf, (0, 3, 1, 2)).astype(np.float32)

    diff = np.max(np.abs(Y_np - Y_tf_nchw))
    print(f"[verify] max|Y_np - Y_tf| = {diff:.6g}")
    print("[verify] OK (within tolerance)" if diff <= atol else "[verify] NG")


def forward_cnn(X, y, Wc, bc, W, b, stride=1, pad=1):
    Zc, conv_cache = conv2d_forward_nchw(X, Wc, bc, stride=stride, pad=pad)
    Ac = relu(Zc)
    N = X.shape[0]
    feat = Ac.reshape(N, -1)

    Z = feat @ W + b
    P = softmax(Z)
    Y = one_hot(y, 10)
    loss = cross_entropy(P, Y)
    cache = {
        "X": X,
        "y": y,
        "Y": Y,
        "Zc": Zc,
        "Ac": Ac,
        "feat": feat,
        "Z": Z,
        "P": P,
        "conv_cache": conv_cache,
        "stride": stride,
        "pad": pad,
        "W": W,
    }
    return loss, cache


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
