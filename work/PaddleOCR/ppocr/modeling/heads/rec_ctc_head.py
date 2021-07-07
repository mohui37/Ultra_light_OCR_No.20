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

import math

import paddle
from paddle import ParamAttr, nn
from paddle.nn import functional as F
import numpy as np



def get_para_bias_attr(l2_decay, k, name):
    regularizer = paddle.regularizer.L2Decay(l2_decay)
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = nn.initializer.Uniform(-stdv, stdv)
    weight_attr = ParamAttr(
        regularizer=regularizer, initializer=initializer, name=name + "_w_attr")
    bias_attr = ParamAttr(
        regularizer=regularizer, initializer=initializer, name=name + "_b_attr")
    return [weight_attr, bias_attr]


class CTCHead(nn.Layer):
    def __init__(self, in_channels, out_channels, fc_decay=0.0004, **kwargs):
        super(CTCHead, self).__init__()

        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=fc_decay, k=in_channels, name='ctc_fc')
        self.fc = nn.Linear(
            in_channels,
            out_channels,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name='ctc_fc')


        weight_attr1, bias_attr1 = get_para_bias_attr(
            l2_decay=fc_decay, k=in_channels, name='ctc_fc1')
        self.fc1 = nn.Linear(
            in_channels,
            9,
            weight_attr=weight_attr1,
            bias_attr=bias_attr1,
            name='ctc_fc1')

        weight_attr2, bias_attr2 = get_para_bias_attr(
            l2_decay=fc_decay, k=in_channels, name='ctc_fc2')
        self.fc2 = nn.Linear(
            in_channels,
            9,
            weight_attr=weight_attr2,
            bias_attr=bias_attr2,
            name='ctc_fc2')

        weight_attr3, bias_attr3 = get_para_bias_attr(
            l2_decay=fc_decay, k=in_channels, name='ctc_fc3')
        self.fc3 = nn.Linear(
            in_channels,
            81,
            weight_attr=weight_attr3,
            bias_attr=bias_attr3,
            name='ctc_fc3')

        weight_attr4, bias_attr4 = get_para_bias_attr(
            l2_decay=fc_decay, k=in_channels, name='ctc_fc4')
        self.fc4 = nn.Linear(
            in_channels,
            81,
            weight_attr=weight_attr4,
            bias_attr=bias_attr4,
            name='ctc_fc4')
        
        self.out_channels = out_channels
        self.in_channels = in_channels
    def forward(self, x, labels=None):
        #print( self.in_channels, self.out_channels)
        
        predicts = self.fc(x)

        '''#final_predicts = np.zeros((128,80,6561))
        #predicts1 = self.fc1(x)
        #predicts2 = self.fc2(x)
        predicts3 = self.fc3(x)
        predicts4 = np.array(np.ones((predicts3.shape[0],predicts3.shape[1],predicts3.shape[2])))
        
        predicts4 = paddle.to_tensor(predicts4, dtype='float32')
        #predicts4 = self.fc4(x)
        
        #final_predicts=paddle.multiply(predicts1,predicts2)
        #predicts1=paddle.reshape(predicts1,(predicts1.shape[0],predicts1.shape[1],1,predicts1.shape[2]))
        #predicts2=paddle.reshape(predicts2,(predicts2.shape[0],predicts2.shape[1],predicts2.shape[2],1))
        #final_predicts=paddle.matmul(predicts2,predicts1)

        predicts3=paddle.reshape(predicts3,(predicts3.shape[0],predicts3.shape[1],predicts3.shape[2],1))
        predicts4=paddle.reshape(predicts4,(predicts4.shape[0],predicts4.shape[1],1,predicts4.shape[2]))
        #print(predicts3,predicts4)
        final_predicts2=paddle.matmul(predicts3,predicts4)
        #print(final_predicts2[0,0,1])

        final_predicts = paddle.reshape(predicts,(predicts.shape[0],predicts.shape[1],predicts.shape[2]))
        final_predicts2 = paddle.reshape(final_predicts2,(final_predicts2.shape[0],final_predicts2.shape[1],final_predicts2.shape[2]*final_predicts2.shape[3]))
        #print(final_predicts.shape,final_predicts2.shape)
        #predicts = final_predicts2
        #print(predicts.shape,final_predicts2.shape)
        predicts = paddle.multiply(final_predicts,final_predicts2)
        #print(predicts.shape)
        #predicts = paddle.reshape(predicts,(predicts.shape[0],predicts.shape[1],predicts.shape[2]*predicts.shape[3]))
        #print(predicts.shape)
        '''
        if not self.training:
            predicts = F.softmax(predicts, axis=2)
        #print('head',predicts.shape)
        return predicts
