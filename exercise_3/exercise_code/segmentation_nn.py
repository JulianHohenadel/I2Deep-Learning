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
        self.base = models.deeplabv3_resnet101(pretrained=True, progress=True)

        aux_classifier = list(self.base.aux_classifier.children())
        self.base.aux_classifier = nn.Sequential(*aux_classifier[:-1])
        self.base.aux_classifier.add_module('4', nn.Conv2d(256, num_classes, 1))

        classifier = list(self.base.classifier.children())
        self.base.classifier = nn.Sequential(*classifier[:-1])
        self.base.classifier.add_module('4', nn.Conv2d(256, num_classes, 1))
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
        return self.base(x)['aux']
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
