import torch
import torch as T
import math
import numpy as np

class Conv2dUnit(torch.nn.Module): # Conv + BN + ReLu
    def __init__(self, input_dim, filters, kernels, stride, padding):
        super(Conv2dUnit, self).__init__()
        self.conv = torch.nn.Conv2d(input_dim, filters, kernel_size=kernels, stride=stride, padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(filters)
        self.leakyreLU = torch.nn.LeakyReLU(0.1)

        # 参数初始化。不这么初始化，容易梯度爆炸nan
        self.conv.weight.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, input_dim, kernels[0], kernels[1])))
        self.bn.weight.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, )))
        self.bn.bias.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, )))
        self.bn.running_mean.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, )))
        self.bn.running_var.data = torch.Tensor(np.random.normal(loc=0.0, scale=0.01, size=(filters, )))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyreLU(x)
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, input_dim, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2dUnit(input_dim, filters, (1, 1), stride=1, padding=0)
        self.conv2 = Conv2dUnit(filters, 2*filters, (3, 3), stride=1, padding=1)
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x

class StackResidualBlock(torch.nn.Module):
    def __init__(self, input_dim, filters, n):
        super(StackResidualBlock, self).__init__()
        self.sequential = torch.nn.Sequential()
        for i in range(n):
            self.sequential.add_module('stack_%d' % (i+1,), ResidualBlock(input_dim, filters))
    def forward(self, x):
        for residual_block in self.sequential:
            x = residual_block(x)
        return x

class Darknet(torch.nn.Module):
    def __init__(self, num_classes, initial_filters=32):
        super(Darknet, self).__init__()
        self.num_classes = num_classes
        i32 = initial_filters
        i64 = i32 * 2
        i128 = i32 * 4
        i256 = i32 * 8
        i512 = i32 * 16
        i1024 = i32 * 32

        ''' darknet53部分，这里所有卷积层都没有偏移bias=False '''
        self.conv1 = Conv2dUnit(3, i32, (3, 3), stride=1, padding=1)
        self.conv2 = Conv2dUnit(i32, i64, (3, 3), stride=2, padding=1)
        self.stack_residual_block_1 = StackResidualBlock(i64, i32, n=1)

        self.conv3 = Conv2dUnit(i64, i128, (3, 3), stride=2, padding=1)
        self.stack_residual_block_2 = StackResidualBlock(i128, i64, n=2)

        self.conv4 = Conv2dUnit(i128, i256, (3, 3), stride=2, padding=1)
        self.stack_residual_block_3 = StackResidualBlock(i256, i128, n=8)

        self.conv5 = Conv2dUnit(i256, i512, (3, 3), stride=2, padding=1)
        self.stack_residual_block_4 = StackResidualBlock(i512, i256, n=8)

        self.conv6 = Conv2dUnit(i512, i1024, (3, 3), stride=2, padding=1)
        self.stack_residual_block_5 = StackResidualBlock(i1024, i512, n=4)
        ''' darknet53部分结束 '''

        self.conv53 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
        self.conv54 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
        self.conv55 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)
        self.conv56 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
        self.conv57 = Conv2dUnit(i1024, i512, (1, 1), stride=1, padding=0)

        self.conv58 = Conv2dUnit(i512, i1024, (3, 3), stride=1, padding=1)
        self.conv59 = torch.nn.Conv2d(i1024, 3*(num_classes + 5), kernel_size=(1, 1))

        self.conv60 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
        self.upsample1 = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.conv61 = Conv2dUnit(i256+i512, i256, (1, 1), stride=1, padding=0)
        self.conv62 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
        self.conv63 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)
        self.conv64 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
        self.conv65 = Conv2dUnit(i512, i256, (1, 1), stride=1, padding=0)

        self.conv66 = Conv2dUnit(i256, i512, (3, 3), stride=1, padding=1)
        self.conv67 = torch.nn.Conv2d(i512, 3*(num_classes + 5), kernel_size=(1, 1))

        self.conv68 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        self.upsample2 = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.conv69 = Conv2dUnit(i128+i256, i128, (1, 1), stride=1, padding=0)
        self.conv70 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)
        self.conv71 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        self.conv72 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)
        self.conv73 = Conv2dUnit(i256, i128, (1, 1), stride=1, padding=0)
        self.conv74 = Conv2dUnit(i128, i256, (3, 3), stride=1, padding=1)

        self.conv75 = torch.nn.Conv2d(i256, 3*(num_classes + 5), kernel_size=(1, 1))

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.stack_residual_block_1(x)
        x = self.conv3(x)
        x = self.stack_residual_block_2(x)
        x = self.conv4(x)
        act11 = self.stack_residual_block_3(x)
        x = self.conv5(act11)
        act19 = self.stack_residual_block_4(x)
        x = self.conv6(act19)
        act23 = self.stack_residual_block_5(x)

        x = self.conv53(act23)
        x = self.conv54(x)
        x = self.conv55(x)
        x = self.conv56(x)
        lkrelu57 = self.conv57(x)

        x = self.conv58(lkrelu57)
        y1 = self.conv59(x)
        y1 = y1.view(y1.size(0), 3, (self.num_classes + 5), y1.size(2), y1.size(3))  # reshape

        x = self.conv60(lkrelu57)
        x = self.upsample1(x)
        x = torch.cat((x, act19), dim=1)

        x = self.conv61(x)
        x = self.conv62(x)
        x = self.conv63(x)
        x = self.conv64(x)
        lkrelu64 = self.conv65(x)

        x = self.conv66(lkrelu64)
        y2 = self.conv67(x)
        y2 = y2.view(y2.size(0), 3, (self.num_classes + 5), y2.size(2), y2.size(3))  # reshape

        x = self.conv68(lkrelu64)
        x = self.upsample2(x)
        x = torch.cat((x, act11), dim=1)

        x = self.conv69(x)
        x = self.conv70(x)
        x = self.conv71(x)
        x = self.conv72(x)
        x = self.conv73(x)
        x = self.conv74(x)
        y3 = self.conv75(x)
        y3 = y3.view(y3.size(0), 3, (self.num_classes + 5), y3.size(2), y3.size(3))  # reshape

        # 相当于numpy的transpose()，交换下标
        y1 = y1.permute(0, 3, 4, 1, 2)
        y2 = y2.permute(0, 3, 4, 1, 2)
        y3 = y3.permute(0, 3, 4, 1, 2)
        return y1, y2, y3