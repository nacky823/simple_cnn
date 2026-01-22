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
