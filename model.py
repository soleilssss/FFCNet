import torch
import torch.nn as nn
from complexLayers import *
from complexFunctions import *
from shift import *
from complexUtils import Sequential_complex


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """
    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = Sequential_complex(
            ComplexConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU(inplace=True),  #inplace = true
            ComplexConv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            ComplexBatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = Sequential_complex()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = Sequential_complex(
                ComplexConv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                ComplexBatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x1, x2):
        x3, x4 = self.residual_function(x1, x2) 
        x1, x2 = complex_relu(x3, x4, inplace=True)
        return x1, x2

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = Sequential_complex(
            ComplexConv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU(inplace=True),
            ComplexConv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU(inplace=True),
            ComplexConv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            ComplexBatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = Sequential_complex()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = Sequential_complex(
                ComplexConv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                ComplexBatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x1, x2):
        return complex_relu(self.residual_function(x1, x2)+ self.shortcut(x1, x2))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=2, inputchannel=1):
        super().__init__()

        self.in_channels = 64
        self.conv1 = ComplexConv2d(in_channels=inputchannel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = ComplexBatchNorm2d(num_features=64)
        self.relu = ComplexReLU(inplace=True)
        self.maxpool = ComplexMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = Complex_AdaptiveAvgPool2d((1, 1))
        self.fc = ComplexLinear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return Sequential_complex(*layers)

    def forward(self, x, target = None):
        xr = x[:,[0,1,2],:,:]
        xi = x[:,[3,4,5],:,:]
        xr, xi = self.conv1(xr, xi)
        xr, xi = self.bn1(xr, xi)
        xr, xi = self.relu(xr, xi)
        xr, xi = self.maxpool(xr, xi)
        xr, xi = self.conv2_x(xr, xi)
        xr, xi = self.conv3_x(xr, xi)
        xr, xi = self.conv4_x(xr, xi)
        xr, xi = self.conv5_x(xr, xi)
        xr, xi = self.avg_pool(xr, xi)
        xr = xr.view(xr.size(0), -1)
        xi = xi.view(xi.size(0), -1)
        xr, xi = self.fc(xr, xi)
        x = torch.sqrt(torch.pow(xr,2)+torch.pow(xi,2))
        return x


def resnet18(num_classes=4, inputchannel=3):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes=4, inputchannel=3)

def resnet34(num_classes=4, inputchannel=3):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes=4, inputchannel=3)

def resnet50(num_classes=4, inputchannel=3):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3],num_classes=4, inputchannel=3)

def resnet101(num_classes=4, inputchannel=3):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3],num_classes=4, inputchannel=3)

def resnet152(num_classes=4, inputchannel=3):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3],num_classes=4, inputchannel=3)



