import os
import tempfile
import numpy as np
from unittest import TestCase
from keras_embed_sim.backend import keras
from keras_embed_sim import EmbeddingRet, EmbeddingSim, get_custom_objects


class TestEmbeddings(TestCase):

    def test_same_return(self):
        input_layer = keras.layers.Input(shape=(None,), name='Input')
        embed, embed_weights = EmbeddingRet(
            input_dim=20,
            output_dim=100,
            mask_zero=True,
            name='Embedding',
        )(input_layer)
        output_layer = EmbeddingSim(
            stop_gradient=True,
            name='Embed-Sim',
        )([embed, embed_weights])
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        model_path = os.path.join(tempfile.gettempdir(), 'test_embed_sim_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
        model.summary(line_length=100)
        batch_inputs = np.random.randint(low=0, high=19, size=(32, 100))
        batch_outputs = model.predict(batch_inputs)
        batch_outputs = np.argmax(batch_outputs, axis=-1)
        self.assertEqual(batch_inputs.tolist(), batch_outputs.tolist())

    def test_no_mask(self):
        input_layer = keras.layers.Input(shape=(None,), name='Input')
        embed, embed_weights = EmbeddingRet(
            input_dim=20,
            output_dim=100,
            name='Embedding',
        )(input_layer)
        output_layer = EmbeddingSim(
            name='Embed-Sim',
        )([embed, embed_weights])
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        model_path = os.path.join(tempfile.gettempdir(), 'test_embed_sim_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
        model.summary(line_length=100)
        batch_inputs = np.random.randint(low=0, high=19, size=(32, 100))
        batch_outputs = model.predict(batch_inputs)
        batch_outputs = np.argmax(batch_outputs, axis=-1)
        self.assertEqual(batch_inputs.tolist(), batch_outputs.tolist())
