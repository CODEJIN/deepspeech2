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
            self.layer_Dict['Layer{}'.format(transformer_Index)] = Multi_Head_Attention(
                num_heads= 1,
                size_per_head= hp_Dict['Transformer']['Size']
                )

    def call(self, inputs, training):
        new_Tensor = inputs
        for transformer_Index in range(hp_Dict['Transformer']['Nums']):
            new_Tensor = self.layer_Dict['Layer{}'.format(transformer_Index)](new_Tensor, new_Tensor, training)                
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

#Refer: https://github.com/google-research/bert/blob/master/modeling.py#L647
class Multi_Head_Attention(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        size_per_head,
        query_activation= None,
        key_activation= None,
        value_activation= None
        ):        
        super(Multi_Head_Attention, self).__init__(name= '')

        self.num_heads = num_heads
        self.size_per_head = size_per_head

        self.layer_Dict = {
            'Query': tf.keras.layers.Dense(
                units= num_heads * size_per_head,
                activation= value_activation,
                use_bias= True,
                kernel_initializer= tf.keras.initializers.TruncatedNormal()
                ),
            'Key': tf.keras.layers.Dense(
                units= num_heads * size_per_head,
                activation= value_activation,
                use_bias= True,
                kernel_initializer= tf.keras.initializers.TruncatedNormal()
                ),
            'Value': tf.keras.layers.Dense(
                units= num_heads * size_per_head,
                activation= value_activation,
                use_bias= True,
                kernel_initializer= tf.keras.initializers.TruncatedNormal()
                )
            }

    def call(self, from_tensor, to_tensor):
        '''
        from_tensor: [Batch, From_Time, From_Dim]
        to_tensor: [Batch, To_Time, To_Dim]
        '''
        query_Tensor = self.layer_Dict['Query'](from_tensor)    #[Batch, From_T, Head * Size]
        key_Tensor = self.layer_Dict['Key'](to_tensor)  #[Batch, To_T, Head * Size]
        value_Tensor = self.layer_Dict['Value'](to_tensor)  #[Batch, To_T, Head * Size]

        query_Tensor = self.reshape_for_score(query_Tensor)   #[Batch, Head, From_T, Size]
        key_Tensor = self.reshape_for_score(key_Tensor)   #[Batch, Head, To_T, Size]
        value_Tensor = self.reshape_for_score(value_Tensor)   #[Batch, Head, To_T, Size]

        attention_Score = tf.matmul(query_Tensor, key_Tensor, transpose_b= True)    #[Batch, Head, From_T, To_T]
        attention_Score *= 1.0 / tf.sqrt(tf.cast(self.size_per_head, tf.float32))
        

    def reshape_to_matrix(self, inputs):
        '''
        inputs: [Batch, Time, Dim]

        return: [Batch*Time, Dim]
        '''
        dim = inputs.get_shape()[-1]
        return tf.reshape(inputs, [-1, dim])

    def reshape_for_score(self, inputs):        
        new_Tensor = tf.reshape(
            inputs,
            [tf.shape(inputs)[0], tf.shape(inputs)[1], self.num_heads, self.size_per_head]
            )
        new_Tensor = tf.transpose(new_Tensor, [0, 2, 1, 3])

        return new_Tensor