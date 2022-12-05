import torch
from torch import nn


class TrainRateLoss(nn.Module):
    def __init__(self):
        super(TrainRateLoss, self).__init__()

    def forward(self, self_informations):
        """ Returns rate in bpp; """
        bpp_rate = torch.mean(self_informations)  # calculating bits per subpixel
        return bpp_rate


class ValidRateLoss(nn.Module):
    def __init__(self):
        super(ValidRateLoss, self).__init__()

    def forward(self, self_informations):
        """ Returns rate in bpp; """
        bpp_rate = torch.mean(self_informations)  # calculating bits per subpixel
        return bpp_rate

    # def forward(self, bitstream, num_pixels):
    #     return len(bitstream) / num_pixels
