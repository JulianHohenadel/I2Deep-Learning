"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision
import torchvision.models.segmentation as models
import torch.nn.functional as F


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        self.num_classes = num_classes
        # self.vgg = models.vgg16(pretrained=True).features
        self.base = models.deeplabv3_resnet101(pretrained=True, progress=True)
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x_input = x
        # print(x.shape) = torch.Size([10, 3, 240, 240])
        x = self.base(x)
        # print(x.shape) = torch.Size([10, 512, 7, 7])
        # x = self.fcn(x)
        # up = nn.Upsample(scale_factor=(240/7), mode='bilinear')
        # x = up(x)
        # print(x.shape)

        # x['out'] = F.interpolate(x['out'], x_input.size()[2:], mode='bilinear')
        return x
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
