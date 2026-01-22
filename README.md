# Simple CNN

MNIST を対象としたシンプルな CNN の実装です．
畳み込み層と全結合層を同時に学習する最小構成にしています．

## Tested environment

+ OS: Ubuntu 22.04 LTS
+ Python: 3.10.12
+ NumPy: 2.2.6
+ TensorFlow: 2.20.0

## Install & Run

```bash
git clone https://github.com/nacky823/simple_cnn.git
cd simple_cnn
```
```bash
python3 simple_cnn.py
```

## Example output

MNIST を読み込み学習し，各エポックの学習データでの正解率と，評価データでの正解率を出力します．

```
MNIST loaded (NCHW):
  X_train: (60000, 1, 28, 28) float32 range (0.0, 1.0)
  y_train: (60000,) int64 labels (0, 9)
  X_test : (10000, 1, 28, 28) float32
  y_test : (10000,) int64
[verify] max|Y_np - Y_tf| = 4.76837e-07
[verify] OK (within tolerance)
epoch  1/5  last_loss 1.2848  train_acc 0.7000  test_acc 0.6200
...
epoch  5/5  last_loss 0.0338  train_acc 0.9950  test_acc 0.7800

Final (subset, trainable conv):
  train_acc: 0.995
  test_acc : 0.78
```

## Model description

入力画像を $x \in \mathbb{R}^{1 \times 28 \times 28}$ とし，バッチサイズ $N$ の入力を
$X \in \mathbb{R}^{N \times 1 \times 28 \times 28}$ とする．
畳み込み層の重みとバイアスを $W_c, b_c$ とすると，特徴マップ上の位置 $(h, w)$ とフィルタ $f$ に対する畳み込みの出力は

$$
Z_c[n,f,h,w] =
\sum_{c}\sum_{i}\sum_{j}
W_c[f,c,i,j]\,
X_p[n,c,h\cdot s+i,\, w\cdot s+j] + b_c[f]
$$

と表せる．$f$ はフィルタ，$c$ は入力チャンネル，$(h,w)$ は特徴マップ上の位置，$(i,j)$ はカーネル内の位置，$X_p$ はパディング後の入力，$s$ はストライドである．

これに活性化関数として ReLU を適用する．

$$
A_c = \mathrm{ReLU}(Z_c)
$$

次に，$A_c$ を平坦化して特徴ベクトル $F$ を作り，全結合層でクラスごとのスコアを得る．

$$
F = \mathrm{flatten}(A_c), \quad Z = F W + b
$$

ソフトマックスで確率 $P$ を計算し，正解ラベル $Y$ との交差エントロピー損失

$$
P = \mathrm{softmax}(Z), \quad
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K Y_{ik}\log(P_{ik})
$$

を最小化するように，$W_c, b_c, W, b$ を更新する．
