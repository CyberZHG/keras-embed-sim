# Keras Embedding Similarity

[![Version](https://img.shields.io/pypi/v/keras-embed-sim.svg)](https://pypi.org/project/keras-embed-sim/)
![License](https://img.shields.io/pypi/l/keras-embed-sim.svg)

\[[中文](https://github.com/CyberZHG/keras-embed-sim/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-embed-sim/blob/master/README.md)\]

计算输出与嵌入层的相似性。

## Install

```bash
pip install keras-embed-sim
```

## Usage

`EmbeddingRet`在一般的嵌入层的基础上会同时返回嵌入的结果和权重，权重可用于`EmbeddingSim`的第二个输入，结果经过softmax，相当于属于某个嵌入的概率。

```python
from tensorflow import keras
from keras_embed_sim import EmbeddingRet, EmbeddingSim

input_layer = keras.layers.Input(shape=(None,), name='Input')

embed, embed_weights = EmbeddingRet(
    input_dim=20,
    output_dim=100,
    mask_zero=True,
)(input_layer)

output_layer = EmbeddingSim()([embed, embed_weights])
```

如果将`EmbeddingSim`中的参数`stop_gradient`设置为`True`，将阻断该层对嵌入权重的梯度传播。
