# Deepspeech 2 replication으로 변경합니다.
# https://github.com/tensorflow/models/blob/master/research/deep_speech/deep_speech_model.py
# 이후 Transformer 적용을 시도해 볼것

import tensorflow as tf
import numpy as np
import json, os, time
from threading import Thread
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime

from Feeder import Feeder
import Modules

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class STT:
    def __init__(self, is_Training= False):
        self.is_Training = is_Training

        self.feeder = Feeder(is_Training= is_Training)
        self.Model_Generate()

    def Model_Generate(self):
        self.model = tf.keras.Sequential([
            Modules.Prenet(),
            Modules.Stacked_Transformer() if hp_Dict['Use_Transformer'] else Modules.RNN(),
            Modules.Postnet(),
            ])
        self.model.build(input_shape=[None, None, hp_Dict['Sound']['Mel_Dim']])

        #optimizer는 @tf.function의 밖에 있어야 함
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate= hp_Dict['Train']['Learning_Rate'],
            beta_1= hp_Dict['Train']['ADAM']['Beta1'],
            beta_2= hp_Dict['Train']['ADAM']['Beta2'],
            epsilon= hp_Dict['Train']['ADAM']['Epsilon'],
            )

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, hp_Dict['Sound']['Mel_Dim']], dtype=tf.float32),
            tf.TensorSpec(shape=[None,], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None,], dtype=tf.int32)
            ],
        autograph= False,
        experimental_relax_shapes= True
        )
    def Train_Step(self, mels, mel_lengths, tokens, token_lengths):
        with tf.GradientTape() as tape:            
            logits = self.model(inputs= mels, training= True)
            loss = tf.reduce_mean(tf.nn.ctc_loss(
                labels= tokens,
                logits= logits,
                label_length= token_lengths,
                logit_length= mel_lengths * tf.shape(logits)[1] // tf.reduce_max(mel_lengths),
                logits_time_major=False,
                blank_index= len(self.feeder.token_Index_Dict),
                ))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, hp_Dict['Sound']['Mel_Dim']], dtype=tf.float32)
            ],
        autograph= False,
        experimental_relax_shapes= True
        )
    def Inference_Step(self, mels):
        logits = self.model(inputs= mels, training= False)
        tokens = tf.math.argmax(logits, axis=-1)
            
        return tokens

    def Restore(self):
        checkpoint_File_Path = os.path.join(hp_Dict['Checkpoint_Path'], 'CHECKPOINT.H5').replace('\\', '/')
        
        if not os.path.exists('{}.index'.format(checkpoint_File_Path)):
            print('There is no checkpoint.')
            return

        self.model.load_weights(checkpoint_File_Path)
        print('Checkpoint \'{}\' is loaded.'.format(checkpoint_File_Path))

    def Train(self):
        def Run_Inference():
            wav_Path_List = []
            with open('Inference_Wav_Path_in_Train.txt', 'r') as f:
                for line in f.readlines():
                    wav_Path_List.append(line.strip())

            self.Inference(wav_Path_List)

        step = 0
        Run_Inference()
        while True:
            start_Time = time.time()
            loss = self.Train_Step(**self.feeder.Get_Train_Pattern())
            step += 1
            display_List = [
                'Time: {:0.3f}'.format(time.time() - start_Time),
                'Step: {}'.format(step),
                'Loss: {:0.5f}'.format(loss)
                ]
            print('\t\t'.join(display_List))

            if step % hp_Dict['Train']['Checkpoint_Save_Timing'] == 0:
                os.makedirs(os.path.join(hp_Dict['Checkpoint_Path']).replace("\\", "/"), exist_ok= True)
                self.model.save_weights(os.path.join(hp_Dict['Checkpoint_Path'], 'CHECKPOINT.H5').replace('\\', '/'))
            
            if step % hp_Dict['Train']['Inference_Timing'] == 0:
                Run_Inference()

    def Inference(self, wav_Path_List, label= None):
        tokens = self.Inference_Step(**self.feeder.Get_Inference_Pattern(wav_Path_List))

        export_Inference_Thread = Thread(
            target= self.Export_Inference,
            args= [
                wav_Path_List,
                tokens,
                label or datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                ]
            )
        export_Inference_Thread.daemon = True
        export_Inference_Thread.start()

    def Export_Inference(self, wav_Path_List, token_List, label):
        os.makedirs(os.path.join(hp_Dict['Inference_Path']).replace("\\", "/"), exist_ok= True)
        
        extract_List= []
        for index, (wav_Path, token) in enumerate(zip(wav_Path_List, token_List)):
            many_to_One_Str, raw_Str = self.Token_to_Str(token.numpy())
            extract_List.append('{}\t{}\t{}\t{}\t{}'.format(label, index, wav_Path, many_to_One_Str, raw_Str))

        with open(os.path.join(hp_Dict['Inference_Path'], 'Inference.txt').replace("\\", "/"), 'a') as f:
            f.write('\n'.join(extract_List) + '\n')

    def Token_to_Str(self, token):
        index_Token_Dict = {index: token for token, index in self.feeder.token_Index_Dict.items()}
        index_Token_Dict.update({len(index_Token_Dict): '/'})

        many_to_One_List = ['/']
        for x in [index_Token_Dict[x] for x in token]:
            if many_to_One_List[-1] != x:
                many_to_One_List.append(x)
        many_to_One_Str = ''.join(many_to_One_List).replace('/', '')
        raw_Str = ''.join([index_Token_Dict[x] for x in token])

        return many_to_One_Str, raw_Str

if __name__ == '__main__':
    new_STT = STT(is_Training= True)
    new_STT.Restore()
    new_STT.Train()
