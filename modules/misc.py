import torch.nn as nn


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global Averge Pooling over the input's spatial dimension"""
        super(GlobalAvgPool2d, self).__init__()

    def foward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
