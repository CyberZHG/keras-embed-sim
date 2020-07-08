# Keras Embedding Similarity

[![Travis](https://travis-ci.org/CyberZHG/keras-embed-sim.svg)](https://travis-ci.org/CyberZHG/keras-embed-sim)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-embed-sim/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-embed-sim)
[![Version](https://img.shields.io/pypi/v/keras-embed-sim.svg)](https://pypi.org/project/keras-embed-sim/)
![Downloads](https://img.shields.io/pypi/dm/keras-embed-sim.svg)
![License](https://img.shields.io/pypi/l/keras-embed-sim.svg)

![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)

\[[中文](https://github.com/CyberZHG/keras-embed-sim/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-embed-sim/blob/master/README.md)\]

Compute the similarity between the outputs and the embeddings.

## Install

```bash
pip install keras-embed-sim
```

## Usage

```python
import keras
from keras_embed_sim import EmbeddingRet, EmbeddingSim

input_layer = keras.layers.Input(shape=(None,), name='Input')

embed, embed_weights = EmbeddingRet(
    input_dim=20,
    output_dim=100,
    mask_zero=True,
)(input_layer)

output_layer = EmbeddingSim()([embed, embed_weights])
```
