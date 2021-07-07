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

__all__ = ['MobileNetV3', 'resmlp','GhostNet']


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
                # k, exp, c,  se,     nl,  s
                [3, 16, 16, False, 'relu', large_stride[0]],
                [3, 64, 24, False, 'relu', (large_stride[1], 1)],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', (large_stride[2], 1)],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hardswish', 1],
                [3, 200, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 480, 112, True, 'hardswish', 1],
                [3, 672, 112, True, 'hardswish', 1],
                [5, 672, 160, True, 'hardswish', (large_stride[3], 1)],
                [5, 960, 160, True, 'hardswish', 1],
                [5, 960, 160, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,  pool
                [3, 16, 16, True, 'relu', (small_stride[0], 1)],
                [3, 72, 24, False, 'relu', (2, 1)],
                [3, 88, 24, False, 'relu', 1],
                [3, 96, 40, True, 'hardswish', (2, 1)],
                [3, 240, 40, True, 'hardswish', 1],
                [3, 240, 40, True, 'hardswish', 1],
                [3, 120, 48, True, 'hardswish', 1],
                [3, 144, 48, True, 'hardswish', 1],
                [3, 288, 96, True, 'hardswish', (2, 1)],
                [3, 576, 96, True, 'hardswish', 1],
                [3, 576, 96, True, 'hardswish', 1],
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
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hardswish',
            name='conv1')
        i = 0
        block_list = []
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in cfg:
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
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

def DW_Conv3x3BNReLU(in_channels,out_channels,stride,groups=1):
    return nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,groups=groups),
            nn.BatchNorm2D(out_channels),
            nn.ReLU6()
    )

class GhostModule(nn.Layer):
    def __init__(self, in_channels,out_channels,s=2, kernel_size=1,stride=1, use_relu=True):
        super(GhostModule, self).__init__()
        intrinsic_channels = out_channels//s
        ghost_channels = intrinsic_channels * (s - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=intrinsic_channels, kernel_size=kernel_size, stride=stride,
                          padding=kernel_size // 2),
            nn.BatchNorm2D(intrinsic_channels),
            nn.ReLU() if use_relu else nn.Sequential()
        )

        self.cheap_op = DW_Conv3x3BNReLU(in_channels=intrinsic_channels, out_channels=ghost_channels, stride=stride,groups=intrinsic_channels)

    def forward(self, x):
        y = self.primary_conv(x)
        z = self.cheap_op(y)
        out = paddle.concat([y, z], axis=1)
        return out

class SqueezeAndExcite(nn.Layer):
    def __init__(self, in_channels, out_channels, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            nn.ReLU6(),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            nn.ReLU6(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        out = self.pool(x)
        out = paddle.reshape(out, [b, -1])
        out = self.SEblock(out)
        out = paddle.reshape(out, [b, c, 1, 1])
        return out * x
class GhostBottleneck(nn.Layer):
    def __init__(self, in_channels,mid_channels, out_channels , kernel_size, stride, use_se, se_kernel_size=1):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        #print(type(self.stride))
        if type(self.stride)==int:
            self.bottleneck = nn.Sequential(
            GhostModule(in_channels=in_channels,out_channels=mid_channels,kernel_size=1,use_relu=True),
            DW_Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=stride,groups=mid_channels) if self.stride>1 else nn.Sequential(),
            SqueezeAndExcite(mid_channels,mid_channels,se_kernel_size) if use_se else nn.Sequential(),
            GhostModule(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, use_relu=False)
            )
            if self.stride>1:
                self.shortcut = DW_Conv3x3BNReLU(in_channels=in_channels, out_channels=out_channels, stride=stride)
            else:
                self.shortcut = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        else:
            self.bottleneck = nn.Sequential(
            GhostModule(in_channels=in_channels,out_channels=mid_channels,kernel_size=1,use_relu=True),
            DW_Conv3x3BNReLU(in_channels=mid_channels, out_channels=mid_channels, stride=stride,groups=mid_channels) if self.stride[0]>1 else nn.Sequential(),
            SqueezeAndExcite(mid_channels,mid_channels,se_kernel_size) if use_se else nn.Sequential(),
            GhostModule(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, use_relu=False)
            )
            if self.stride[0]>1:
                self.shortcut = DW_Conv3x3BNReLU(in_channels=in_channels, out_channels=out_channels, stride=stride)
            else:
                self.shortcut = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        

    def forward(self, x):
        out = self.bottleneck(x)
        residual = self.shortcut(x)
        out += residual
        return out

class GhostNet(nn.Layer):
    def __init__(self, num_classes=1000,in_channels=3,out_channels =320):
        super(GhostNet, self).__init__()
        self.out_channels =out_channels
        self.first_conv = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(16),
            nn.ReLU6(),
        )

        #172万参数版本
        '''self.features = nn.Sequential(
            GhostBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=1, use_se=False),
            GhostBottleneck(in_channels=16, mid_channels=64, out_channels=24, kernel_size=3, stride=(2,1),  use_se=False),
            #GhostBottleneck(in_channels=24, mid_channels=72, out_channels=24, kernel_size=3, stride=1,  use_se=False),
            GhostBottleneck(in_channels=24, mid_channels=72, out_channels=40, kernel_size=3, stride=2, use_se=True, se_kernel_size=28),
            GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=3, stride=1, use_se=True, se_kernel_size=28),
            #GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, use_se=True, se_kernel_size=28),
            GhostBottleneck(in_channels=40, mid_channels=240, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            GhostBottleneck(in_channels=80, mid_channels=200, out_channels=80, kernel_size=3, stride=1, use_se=False),
            #GhostBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            #GhostBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            GhostBottleneck(in_channels=80, mid_channels=480, out_channels=112, kernel_size=3, stride=(1,1), use_se=True, se_kernel_size=14),
            #GhostBottleneck(in_channels=112, mid_channels=672, out_channels=112, kernel_size=3, stride=1, use_se=True, se_kernel_size=14),
            GhostBottleneck(in_channels=112, mid_channels=672, out_channels=160, kernel_size=3, stride=(1,1), use_se=True,se_kernel_size=7),
            #GhostBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, use_se=True,se_kernel_size=7),
            #GhostBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=2, use_se=True,se_kernel_size=7),
        )

        self.last_stage  = nn.Sequential(
            
            #148万参数
            nn.Conv2D(in_channels=160, out_channels=320, kernel_size=(2,1), stride=1),
            nn.BatchNorm2D(320),
            nn.ReLU6(),
            
          
        )
        '''
        #190万参数版本
        '''self.features = nn.Sequential(
            GhostBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=1, use_se=False),
            GhostBottleneck(in_channels=16, mid_channels=64, out_channels=24, kernel_size=3, stride=(2,1),  use_se=False),
            #GhostBottleneck(in_channels=24, mid_channels=72, out_channels=24, kernel_size=3, stride=1,  use_se=False),
            GhostBottleneck(in_channels=24, mid_channels=72, out_channels=40, kernel_size=5, stride=2, use_se=True, se_kernel_size=28),
            GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, use_se=True, se_kernel_size=28),
            #GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, use_se=True, se_kernel_size=28),
            GhostBottleneck(in_channels=40, mid_channels=240, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            GhostBottleneck(in_channels=80, mid_channels=200, out_channels=80, kernel_size=3, stride=1, use_se=False),
            #GhostBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            #GhostBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            GhostBottleneck(in_channels=80, mid_channels=480, out_channels=112, kernel_size=3, stride=(2,1), use_se=True, se_kernel_size=14),
            #GhostBottleneck(in_channels=112, mid_channels=672, out_channels=112, kernel_size=3, stride=1, use_se=True, se_kernel_size=14),
            GhostBottleneck(in_channels=112, mid_channels=672, out_channels=160, kernel_size=5, stride=(2,1), use_se=True,se_kernel_size=7),
            #GhostBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, use_se=True,se_kernel_size=7),
            #GhostBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=2, use_se=True,se_kernel_size=7),
        )
        self.last_stage  = nn.Sequential(
            
            #148万参数
            nn.Conv2D(in_channels=160, out_channels=320, kernel_size=1, stride=1),
            nn.BatchNorm2D(320),
            nn.ReLU6(),
            
          
        )
        '''

        #148万参数版本
        '''self.features = nn.Sequential(
            GhostBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=1, use_se=False),
            GhostBottleneck(in_channels=16, mid_channels=64, out_channels=24, kernel_size=3, stride=(2,1),  use_se=False),
            #GhostBottleneck(in_channels=24, mid_channels=72, out_channels=24, kernel_size=3, stride=1,  use_se=False),
            GhostBottleneck(in_channels=24, mid_channels=72, out_channels=40, kernel_size=5, stride=2, use_se=True, se_kernel_size=28),
            #GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, use_se=True, se_kernel_size=28),
            #GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, use_se=True, se_kernel_size=28),
            GhostBottleneck(in_channels=40, mid_channels=240, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            GhostBottleneck(in_channels=80, mid_channels=200, out_channels=80, kernel_size=3, stride=1, use_se=False),
            #GhostBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            #GhostBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            GhostBottleneck(in_channels=80, mid_channels=480, out_channels=112, kernel_size=3, stride=(2,1), use_se=True, se_kernel_size=14),
            #GhostBottleneck(in_channels=112, mid_channels=672, out_channels=112, kernel_size=3, stride=1, use_se=True, se_kernel_size=14),
            #GhostBottleneck(in_channels=112, mid_channels=672, out_channels=160, kernel_size=5, stride=(2,1), use_se=True,se_kernel_size=7),
            #GhostBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, use_se=True,se_kernel_size=7),
            #GhostBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=2, use_se=True,se_kernel_size=7),
        )
        self.last_stage  = nn.Sequential(
            
            #148万参数
            nn.Conv2D(in_channels=112, out_channels=320, kernel_size=1, stride=1),
            nn.BatchNorm2D(320),
            nn.ReLU6(),
            
          
        )
        '''

        #190_ksize=3
        '''self.features = nn.Sequential(
            GhostBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=1, use_se=False),
            GhostBottleneck(in_channels=16, mid_channels=64, out_channels=24, kernel_size=3, stride=(2,1),  use_se=False),
            #GhostBottleneck(in_channels=24, mid_channels=72, out_channels=24, kernel_size=3, stride=1,  use_se=False),
            GhostBottleneck(in_channels=24, mid_channels=72, out_channels=40, kernel_size=3, stride=2, use_se=True, se_kernel_size=28),
            GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=3, stride=1, use_se=True, se_kernel_size=28),
            #GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, use_se=True, se_kernel_size=28),
            GhostBottleneck(in_channels=40, mid_channels=240, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            GhostBottleneck(in_channels=80, mid_channels=200, out_channels=80, kernel_size=3, stride=1, use_se=False),
            #GhostBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            #GhostBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            GhostBottleneck(in_channels=80, mid_channels=480, out_channels=112, kernel_size=3, stride=(2,1), use_se=True, se_kernel_size=14),
            #GhostBottleneck(in_channels=112, mid_channels=672, out_channels=112, kernel_size=3, stride=1, use_se=True, se_kernel_size=14),
            GhostBottleneck(in_channels=112, mid_channels=672, out_channels=160, kernel_size=3, stride=(2,1), use_se=True,se_kernel_size=7),
            #GhostBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, use_se=True,se_kernel_size=7),
            #GhostBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=2, use_se=True,se_kernel_size=7),
        )
        self.last_stage  = nn.Sequential(
            
            #148万参数
            nn.Conv2D(in_channels=160, out_channels=320, kernel_size=1, stride=1),
            nn.BatchNorm2D(320),
            nn.ReLU6(),
            
          
        )
        '''
        self.features = nn.Sequential(
            GhostBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=1, use_se=False),
            GhostBottleneck(in_channels=16, mid_channels=64, out_channels=24, kernel_size=3, stride=(2,1),  use_se=False),
            #GhostBottleneck(in_channels=24, mid_channels=72, out_channels=24, kernel_size=3, stride=1,  use_se=False),
            GhostBottleneck(in_channels=24, mid_channels=72, out_channels=40, kernel_size=3, stride=2, use_se=True, se_kernel_size=28),
            GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=3, stride=1, use_se=True, se_kernel_size=28),
            #GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, kernel_size=5, stride=1, use_se=True, se_kernel_size=28),
            GhostBottleneck(in_channels=40, mid_channels=240, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            GhostBottleneck(in_channels=80, mid_channels=200, out_channels=80, kernel_size=3, stride=1, use_se=False),
            #GhostBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            #GhostBottleneck(in_channels=80, mid_channels=184, out_channels=80, kernel_size=3, stride=(2,1), use_se=False),
            GhostBottleneck(in_channels=80, mid_channels=480, out_channels=112, kernel_size=3, stride=(2,1), use_se=True, se_kernel_size=14),
            #GhostBottleneck(in_channels=112, mid_channels=672, out_channels=112, kernel_size=3, stride=1, use_se=True, se_kernel_size=14),
            GhostBottleneck(in_channels=112, mid_channels=672, out_channels=160, kernel_size=3, stride=(2,1), use_se=True,se_kernel_size=7),
            #GhostBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, use_se=True,se_kernel_size=7),
            #GhostBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=2, use_se=True,se_kernel_size=7),
        )
        self.last_stage  = nn.Sequential(
            
            #148万参数
            nn.Conv2D(in_channels=160, out_channels=320, kernel_size=1, stride=1),
            nn.BatchNorm2D(320),
            nn.ReLU6(),
            
          
        )
        

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal(m.weight)
                nn.initializer.constant(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.initializer.constant(m.weight, 1)
                nn.initializer.constant(m.bias, 0) 

    def forward(self, x):
        x = self.first_conv(x)
        #print(x.shape)
        x = self.features(x)
        #print(x.shape)
        x= self.last_stage(x)
        #x = paddle.reshape(x, [x.shape[0], -1])
        #print(x.shape)
        return x
def get_parameter_number(net): 
    total_num = sum(p.numel() for p in net.parameters()) 
    trainable_num = sum(p.numel() for p in net.parameters() if not p.stop_gradient) 
    return {'Total': total_num,'Train': trainable_num}
if __name__ == '__main__':
    model = GhostNet()
    #print(model)
    print(get_parameter_number(model))
    input = paddle.randn([1, 3, 32, 320])
    out = model(input)
    print(out.shape)