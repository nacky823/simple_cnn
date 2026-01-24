# SPDX-FileCopyrightText: 2026 Yuki NAGAKI youjiyongmu4@gmail.com
# SPDX-License-Identifier: BSD-3-Clause
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


def backward_cnn(cache, Wc, bc, W, b):
    X = cache["X"]
    Y = cache["Y"]
    P = cache["P"]
    Zc = cache["Zc"]
    Ac = cache["Ac"]
    feat = cache["feat"]
    conv_cache = cache["conv_cache"]
    N = X.shape[0]

    dZ = (P - Y) / N
    dW = feat.T @ dZ
    db = np.sum(dZ, axis=0)
    dfeat = dZ @ W.T
    dAc = dfeat.reshape(Ac.shape)
    dZc = relu_backward(dAc, Zc)
    dX, dWc, dbc = conv2d_backward_nchw(dZc, conv_cache)
    return dWc, dbc, dW, db, dX


def train_cnn(
    X_train, y_train, X_test, y_test,
    F=4, HH=3, WW=3, stride=1, pad=1,
    lr=0.1, batch_size=32, epochs=5, seed=0
):
    rng = np.random.default_rng(seed)
    N, C, H, W_in = X_train.shape
    K = 10
    Wc = (rng.standard_normal((F, C, HH, WW)) * np.sqrt(2.0 / (C * HH * WW))).astype(np.float32)
    bc = np.zeros((F,), dtype=np.float32)

    H_out = (H + 2 * pad - HH) // stride + 1
    W_out = (W_in + 2 * pad - WW) // stride + 1
    Dfeat = F * H_out * W_out

    W = (rng.standard_normal((Dfeat, K)) * np.sqrt(2.0 / Dfeat)).astype(np.float32)
    b = np.zeros((K,), dtype=np.float32)
    steps_per_epoch = (N + batch_size - 1) // batch_size

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(N)
        last_loss = None
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size: (step + 1) * batch_size]
            Xb = X_train[idx]
            yb = y_train[idx]
            loss, cache = forward_cnn(Xb, yb, Wc, bc, W, b, stride=stride, pad=pad)
            last_loss = loss
            dWc, dbc, dW, db, _ = backward_cnn(cache, Wc, bc, W, b)
            Wc -= lr * dWc
            bc -= lr * dbc
            W -= lr * dW
            b -= lr * db
        tr_logits = predict_logits(X_train, Wc, bc, W, b, stride=stride, pad=pad)
        te_logits = predict_logits(X_test, Wc, bc, W, b, stride=stride, pad=pad)
        tr_acc = accuracy_from_logits(tr_logits, y_train)
        te_acc = accuracy_from_logits(te_logits, y_test)
        print(f"epoch {epoch:2d}/{epochs}  last_loss {last_loss:.4f}  train_acc {tr_acc:.4f}  test_acc {te_acc:.4f}")
    return Wc, bc, W, b


def predict_logits(X, Wc, bc, W, b, stride=1, pad=1):
    Zc, _ = conv2d_forward_nchw(X, Wc, bc, stride=stride, pad=pad)
    Ac = relu(Zc)
    feat = Ac.reshape(Ac.shape[0], -1)
    return feat @ W + b


def save_prediction_grid(X, y_true, logits, path, max_images=25, ncols=5):
    import os
    import matplotlib.pyplot as plt

    X_img = X[:max_images, 0]
    y_true = y_true[:max_images]
    y_pred = np.argmax(logits[:max_images], axis=1)
    n = X_img.shape[0]
    ncols = min(ncols, n) if n > 0 else 1
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(X_img[i], cmap="gray", vmin=0.0, vmax=1.0)
        correct = (y_pred[i] == y_true[i])
        color = "green" if correct else "red"
        ax.set_title(f"p={y_pred[i]} t={y_true[i]}", color=color, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_misclassified_grid(X, y_true, logits, path, max_images=25, ncols=5):
    import os
    import matplotlib.pyplot as plt

    y_pred = np.argmax(logits, axis=1)
    miss_idx = np.where(y_pred != y_true)[0]
    if miss_idx.size == 0:
        print("[viz] no misclassifications to plot")
        return
    miss_idx = miss_idx[:max_images]
    X_img = X[miss_idx, 0]
    y_true_m = y_true[miss_idx]
    y_pred_m = y_pred[miss_idx]

    n = X_img.shape[0]
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(X_img[i], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"p={y_pred_m[i]} t={y_true_m[i]}", color="red", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_first_layer_filters(Wc, path, ncols=8):
    import os
    import matplotlib.pyplot as plt

    F, C, HH, WW = Wc.shape
    assert C == 1, "only single-channel filters are supported"
    n = F
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.6))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        filt = Wc[i, 0]
        fmin, fmax = float(filt.min()), float(filt.max())
        if fmax > fmin:
            img = (filt - fmin) / (fmax - fmin)
        else:
            img = np.zeros_like(filt)
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"f{i}", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_feature_maps(X, Wc, bc, path, sample_indices=None, num_samples=5, stride=1, pad=1, ncols=5):
    import os
    import matplotlib.pyplot as plt

    if sample_indices is None:
        sample_indices = list(range(min(num_samples, X.shape[0])))
    else:
        sample_indices = list(sample_indices)[:num_samples]

    Xb = X[sample_indices]
    Zc, _ = conv2d_forward_nchw(Xb, Wc, bc, stride=stride, pad=pad)
    Ac = relu(Zc)  # (N, F, H, W)

    N = Ac.shape[0]
    F = Ac.shape[1]
    ncols = min(ncols, N)
    nrows = F + 1  # input + feature maps

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.0))
    axes = np.array(axes).reshape(nrows, ncols)
    for c in range(ncols):
        for r in range(nrows):
            ax = axes[r, c]
            ax.axis("off")
            if r == 0:
                ax.imshow(Xb[c, 0], cmap="gray", vmin=0.0, vmax=1.0)
                ax.set_title(f"input[{sample_indices[c]}]", fontsize=8)
                continue
            fmap = Ac[c, r - 1]
            fmin, fmax = float(fmap.min()), float(fmap.max())
            if fmax > fmin:
                img = (fmap - fmin) / (fmax - fmin)
            else:
                img = np.zeros_like(fmap)
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            if c == 0:
                ax.set_ylabel(f"f{r - 1}", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist_nchw()
    print("MNIST loaded (NCHW):")
    print("  X_train:", X_train.shape, X_train.dtype, "range", (float(X_train.min()), float(X_train.max())))
    print("  y_train:", y_train.shape, y_train.dtype, "labels", (int(y_train.min()), int(y_train.max())))
    print("  X_test :", X_test.shape, X_test.dtype)
    print("  y_test :", y_test.shape, y_test.dtype)

    X_train_s = X_train[:200]
    y_train_s = y_train[:200]
    X_test_s = X_test[:100]
    y_test_s = y_test[:100]

    F = 4
    HH = 3
    WW = 3
    stride = 1
    pad = 1

    rng = np.random.default_rng(0)
    Wc_tmp = (rng.standard_normal((F, 1, HH, WW)) * np.sqrt(2.0 / (1 * HH * WW))).astype(np.float32)
    bc_tmp = np.zeros((F,), dtype=np.float32)
    verify_conv_forward_with_tf(X_train_s[:8], Wc_tmp, bc_tmp, stride=stride, pad=pad, atol=1e-4)

    Wc, bc, W, b = train_cnn(
        X_train_s, y_train_s, X_test_s, y_test_s,
        F=F, HH=HH, WW=WW, stride=stride, pad=pad,
        lr=0.1, batch_size=20, epochs=5, seed=0
    )

    tr_logits = predict_logits(X_train_s, Wc, bc, W, b, stride=stride, pad=pad)
    te_logits = predict_logits(X_test_s, Wc, bc, W, b, stride=stride, pad=pad)
    print("\nFinal (subset, trainable conv):")
    print("  train_acc:", accuracy_from_logits(tr_logits, y_train_s))
    print("  test_acc :", accuracy_from_logits(te_logits, y_test_s))
