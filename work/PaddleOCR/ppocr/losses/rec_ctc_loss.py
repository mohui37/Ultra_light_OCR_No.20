# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from operator import le

import numpy as np 
import paddle
from paddle import nn
from paddle import nn
from paddle.nn import functional as F



class CTCLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')

    def __call__(self, predicts, batch):
        
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = paddle.to_tensor([N] * B, dtype='int64')
        labels = batch[1].astype("int32")
        #print(predicts.shape,labels.shape)
        label_lengths = batch[2].astype('int64')
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        loss = loss.mean()  # sum
        return {'loss': loss}

class focal_CTCLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(focal_CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')

    def __call__(self, predicts, batch,alpha=0.5,gamma=2.0):
        
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = paddle.to_tensor([N] * B, dtype='int64')
        labels = batch[1].astype("int32")
        #print(predicts.shape,labels.shape)
        label_lengths = batch[2].astype('int64')
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        p = paddle.exp(-loss)
        loss= paddle.multiply(paddle.multiply(paddle.to_tensor(alpha),paddle.pow((1-p),paddle.to_tensor(gamma))),loss) #((alpha)*((1-p)**gamma)*(ctc_loss))
        loss = loss.mean()  # sum
        return {'loss': loss}

class aceLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(aceLoss, self).__init__()
        #self.loss_func = nn.ACELoss(blank=0, reduction='none')

    def __call__(self, predicts, batch):
        #print(predicts.shape)
        b, n, c = predicts.shape
        predicts = F.softmax(predicts, axis=2)
        #print(predicts[0],final_label[0])
        predicts = predicts+ 1e-10 
        predicts = paddle.sum(predicts,1)
        predicts = predicts/n

        label = batch[1].astype("int32")
        final_label=np.zeros((b, c))
        lebel_length=batch[2].astype("int32")
        for i in range(len(label)):
            #print(label[i])
            for j in range(len(lebel_length[i])):
                final_label[i][label[i][j]]+=1
        final_label= final_label/n
        final_label = paddle.to_tensor(final_label)

        #no_batch
        #loss = (-paddle.sum(paddle.log(predicts) * final_label)) / b
        #batch
        loss = (-paddle.sum(paddle.log(predicts) * final_label)) 
        #print(predicts[0],final_label[0])
        #print(paddle.log(predicts))
        #print(paddle.log(predicts) * final_label)
        #print(-paddle.sum(paddle.log(predicts) * final_label))
        #print(loss)
        return {'loss': loss}