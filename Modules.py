# Keras에서 layer dict의 경우 str이외의 key를 사용할 수 없음

import tensorflow as tf
import numpy as np
import json

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)
with open(hp_Dict['Token_JSON_Path'], 'r') as f:
    token_Index_Dict = json.load(f)


class Prenet(tf.keras.Model):
    def __init__(self):
        super(Prenet, self).__init__(name= '')

        self.layer_Dict = {}
        for prenet_Index in range(hp_Dict['Prenet']['Nums']):
            self.layer_Dict['Layer{}_Padding'.format(prenet_Index)] =  tf.keras.layers.ZeroPadding1D(
                padding= hp_Dict['Prenet']['Conv']['Kernel'][prenet_Index] // 2
                )
            self.layer_Dict['Layer{}_Conv1D'.format(prenet_Index)] = tf.keras.layers.Conv1D(
                filters= hp_Dict['Prenet']['Conv']['Channel'][prenet_Index],
                kernel_size= hp_Dict['Prenet']['Conv']['Kernel'][prenet_Index],
                strides= hp_Dict['Prenet']['Conv']['Strides'][prenet_Index],
                padding= 'valid',
                activation= tf.nn.relu6,
                use_bias= False,
                )
            self.layer_Dict['Layer{}_BatchNormalization'.format(prenet_Index)]= tf.keras.layers.BatchNormalization(
                momentum= hp_Dict['Batch_Normalization']['Momentum'],
                epsilon= hp_Dict['Batch_Normalization']['Epsilon'],                
                )
            self.layer_Dict['Layer{}_Dropout'.format(prenet_Index)]= tf.keras.layers.Dropout(
                rate= hp_Dict['Prenet']['Dropout']['Rate'],
                )

    def call(self, inputs, training= False):
        new_Tensor = inputs
        for prenet_Index in range(hp_Dict['Prenet']['Nums']):
            new_Tensor = self.layer_Dict['Layer{}_Padding'.format(prenet_Index)](new_Tensor)
            new_Tensor = self.layer_Dict['Layer{}_Conv1D'.format(prenet_Index)](new_Tensor)            
            new_Tensor = self.layer_Dict['Layer{}_BatchNormalization'.format(prenet_Index)](new_Tensor, training= training)
            new_Tensor = self.layer_Dict['Layer{}_Dropout'.format(prenet_Index)](new_Tensor, training= training)

        return new_Tensor

class RNN(tf.keras.Model):
    def __init__(self):
        super(RNN, self).__init__(name='')

        self.layer_Dict = {}
        for rnn_Index in range(hp_Dict['BiLSTM']['Nums']):
            self.layer_Dict['Layer{}'.format(rnn_Index)] = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units= hp_Dict['BiLSTM']['Uni_Direction_Cell_Size'],
                    activation='sigmoid',
                    return_sequences=True
                    )
                )

    def call(self, inputs):
        new_Tensor = inputs
        for rnn_Index in range(hp_Dict['BiLSTM']['Nums']):
            new_Tensor = self.layer_Dict['Layer{}'.format(rnn_Index)](new_Tensor)

        return new_Tensor

class Stacked_Transformer(tf.keras.Model):
    def __init__(self):
        super(Stacked_Transformer, self).__init__(name='')

        self.layer_Dict = {}
        for transformer_Index in range(hp_Dict['Transformer']['Nums']):
            for input_Type in ['Query', 'Key']:
                self.layer_Dict['Layer{}_{}'.format(transformer_Index, input_Type)] = tf.keras.layers.Dense(
                    units= hp_Dict['Transformer']['Size'],
                    use_bias= True,
                    )
            self.layer_Dict['Layer{}_Attention'.format(transformer_Index)] = tf.keras.layers.Attention(
                use_scale= True,
                causal= False
                )

    def call(self, inputs):
        new_Tensor = inputs
        for transformer_Index in range(hp_Dict['Transformer']['Nums']):
            new_Tensor = self.layer_Dict['Layer{}_Attention'.format(transformer_Index)]([
                self.layer_Dict['Layer{}_Query'.format(transformer_Index)](inputs),
                self.layer_Dict['Layer{}_Key'.format(transformer_Index)](inputs),
                ])
        return new_Tensor

class Postnet(tf.keras.layers.Layer):   #it is same to dense now.
    def __init__(self):
        super(Postnet, self).__init__(name= '')

        self.layer = tf.keras.layers.Dense(
            units= len(token_Index_Dict) + 1,
            use_bias= False,
            )

    def call(self, inputs):        
        return self.layer(inputs)