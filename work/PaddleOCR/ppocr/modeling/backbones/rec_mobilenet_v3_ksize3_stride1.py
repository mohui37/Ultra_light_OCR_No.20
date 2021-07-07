# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from paddle import nn
import paddle
from ppocr.modeling.backbones.det_mobilenet_v3 import ResidualUnit, ConvBNLayer, make_divisible
from paddle import ParamAttr

__all__ = ['MobileNetV3', 'resmlp']


class MobileNetV3(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 model_name='small',
                 scale=0.5,
                 large_stride=None,
                 small_stride=None,
                 **kwargs):
        super(MobileNetV3, self).__init__()
        if small_stride is None:
            small_stride = [2, 2, 2, 2]
        if large_stride is None:
            large_stride = [1, 2, 2, 2]

        assert isinstance(large_stride, list), "large_stride type must " \
                                               "be list but got {}".format(type(large_stride))
        assert isinstance(small_stride, list), "small_stride type must " \
                                               "be list but got {}".format(type(small_stride))
        assert len(large_stride) == 4, "large_stride length must be " \
                                       "4 but got {}".format(len(large_stride))
        assert len(small_stride) == 4, "small_stride length must be " \
                                       "4 but got {}".format(len(small_stride))

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,  pool
                [3, 16, 16, False, 'relu', large_stride[0], False],
                [3, 64, 24, False, 'relu', (large_stride[1], 1), False],
                [3, 72, 24, False, 'relu', 1, False],
                [5, 72, 40, True, 'relu', (large_stride[2], 1), False],
                [5, 120, 40, True, 'relu', 1, False],
                [5, 120, 40, True, 'relu', 1, False],
                [3, 240, 80, False, 'hardswish', 1, False],
                [3, 200, 80, False, 'hardswish', 1, False],
                [3, 184, 80, False, 'hardswish', 1, False],
                [3, 184, 80, False, 'hardswish', 1, False],
                [3, 480, 112, True, 'hardswish', 1, False],
                [3, 672, 112, True, 'hardswish', 1, False],
                [5, 672, 160, True, 'hardswish', (large_stride[3], 1), False],
                [5, 960, 160, True, 'hardswish', 1, False],
                [5, 960, 160, True, 'hardswish', 1, False],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,  pool
                [3, 16, 16, True, 'relu', (1, 1), False],
                [3, 72, 24, False, 'relu', (1, 1), False],
                [3, 88, 24, False, 'relu', 1, False],
                [3, 96, 40, True, 'hardswish', (1, 1), False],
                [3, 240, 40, True, 'hardswish', 1, False],
                [3, 240, 40, True, 'hardswish', 1, False],
                [3, 120, 48, True, 'hardswish', 1, False],
                [3, 144, 48, True, 'hardswish', 1, False],
                [3, 288, 96, True, 'hardswish', (1, 1), False],
                [3, 576, 96, True, 'hardswish', 1, False],
                [3, 576, 96, True, 'hardswish', 1, False],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scales are {} but input scale is {}".format(supported_scale, scale)

        inplanes = 16
        # conv1
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv1')
        i = 0
        block_list = []
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s, p) in cfg:
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    pool=p,
                    act=nl,
                    name='conv' + str(i + 2)))
            inplanes = make_divisible(scale * c)
            i += 1
        self.blocks = nn.Sequential(*block_list)

        self.conv2 = ConvBNLayer(
            in_channels=inplanes,
            out_channels=make_divisible(scale * cls_ch_squeeze),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv_last')

        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.out_channels = make_divisible(scale * cls_ch_squeeze)
        
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        #print(x.shape)
        x = self.conv2(x)
        x = self.pool(x)
        

        return x

class Aff(nn.Layer): 
    def __init__(self, dim):
        super().__init__()

        alpha = self.create_parameter(shape=[1, 1, dim], default_initializer=nn.initializer.Constant(value=1))
        beta = self.create_parameter(shape=[1, 1, dim], default_initializer=nn.initializer.Constant(value=0))

        self.add_parameter("alpha", alpha)
        self.add_parameter("beta", beta)

    def forward(self, x):
        return self.beta + paddle.multiply(self.alpha, x)

class FeedForward(nn.Layer):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MLPblock(nn.Layer):

    def __init__(self, dim, num_patch, mlp_dim, dropout = 0., init_values=1e-4):
        super().__init__()

        self.pre_affine = Aff(dim)
        self.token_mix = nn.Linear(num_patch, num_patch)
        
        self.ff = FeedForward(dim, mlp_dim, dropout)
        
        self.post_affine = Aff(dim)
        gamma_1 = self.create_parameter(shape=[dim], default_initializer=nn.initializer.Constant(init_values))
        gamma_2 = self.create_parameter(shape=[dim], default_initializer=nn.initializer.Constant(init_values))
        self.add_parameter("gamma_1", gamma_1)
        self.add_parameter("gamma_2", gamma_2)

    def forward(self, x):
        x = self.pre_affine(x)
        x = x + self.gamma_1 * paddle.transpose(self.token_mix(paddle.transpose(x, perm=[0, 2, 1])),perm=[0, 2, 1])
        x = self.post_affine(x)
        y = self.ff(x)
        y = self.gamma_2 * y
        x = x + y
        return x

class resmlp(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 dim=512,
                 patch_size=(8,32),
                 depth=12,
                 **kwargs):
        super(resmlp, self).__init__()

        block_list = []
        for i in range(depth):
            block_list.append(MLPblock(dim, num_patch=40, mlp_dim=dim*4))
        self.mlp_blocks = nn.Sequential(*block_list)

        
        self.to_patch_embedding = nn.Conv2D(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            weight_attr=ParamAttr(name='firstconv' + '_weights'),
            bias_attr=False)
        self.affine = Aff(dim)
        self.out_channels = dim
    def forward(self, x):

        x = self.to_patch_embedding(x)
        x = paddle.transpose(x, perm=[0, 2, 3, 1])
        x = paddle.reshape(x, [x.shape[0], x.shape[1]*x.shape[2], x.shape[3]])

        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)

        x = self.affine(x)
        
        x  = paddle.transpose(x, perm=[0, 2, 1])
        x = paddle.reshape(x, [x.shape[0], x.shape[1], 1, x.shape[2]])
        #print(x.shape)
        return x