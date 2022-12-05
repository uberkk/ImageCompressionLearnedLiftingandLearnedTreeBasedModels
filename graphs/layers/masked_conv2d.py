# https://www.codeproject.com/Articles/5061271/PixelCNN-in-Autoregressive-Models
from torch import nn


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ('A', 'B')
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        if kW > 1:
            self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        elif kW == 1 and mask_type == 'A':
            self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0  # don't forget to set 1x1 mask to zero if 'A'
        if kH > 1:
            self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class MaskedConv2d_zhat_x(nn.Conv2d):
    def __init__(self, mask_type, in_channels_x, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ('A')
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        if kW > 1:
            self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        elif kW == 1 and mask_type == 'A':
            self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0  # don't forget to cet 1x1 mask to zero if 'A'
        if kH > 1:
            self.mask[:, :, kH // 2 + 1:] = 0
        in_ch_x, out_ch, stride_ks = 3, 128, 1
        for name, value in kwargs.items():
            if name == 'stride':
                stride_ks = value
            elif name == 'out_channels':
                out_ch = value
        self.conv2d_blkbyblk = nn.Conv2d(in_channels=in_channels_x, out_channels=out_ch, stride=stride_ks, kernel_size=stride_ks)

    def forward(self, zhat, x):
        self.weight.data *= self.mask
        zhat_result = super(MaskedConv2d_zhat_x, self).forward(zhat)
        x_result = self.conv2d_blkbyblk.forward(x)
        return zhat_result + x_result


class Conv2dChIncremental(nn.Conv2d):
    """ Describe class and constraints...
    Each output channel sees only input channels upto and including that channel when in_channels=out_channels.
    When in_channels<out_channels or in_channels>out_channels, in which case they must be integer K multiples of each
    other. When in_channels<out_channels, each ch i*K:i*K+K see input channels 0:i+1.
    When in_channels>out_channels, each ch i:i+1 see input channels 0:i*K+K.
    grps determines grouping for in_channels when oC >= iC and for out_channels when oC < iC
    E.g.    Conv2dChIncremental(grps=1, in_channels=4,  out_channels=32)
            Conv2dChIncremental(grps=8, in_channels=32, out_channels=32)
            Conv2dChIncremental(grps=8, in_channels=32, out_channels=32)
            Conv2dChIncremental(grps=1, in_channels=32, out_channels=2*4)
            Then take cahnnels 0:2(4-1) for 3 mu+sigma channels when xe=1, xo=3
    """
    def __init__(self, grps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # get in out channel sizes and set parameters
        oC, iC, kH, kW = self.weight.size()
        assert (oC % iC == 0) or (iC % oC == 0)  # iC and oC must be integer multiples of each other;
        if oC >= iC:
            ch_mode = 'up'
            assert iC % grps == 0
        else:
            ch_mode = 'dn'
            assert oC % grps == 0
        # prepare mask for masking convolution weights
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(0)
        if ch_mode == 'up':
            K = oC // iC
            for i in range(0, iC, grps):
                self.mask[i*K:(i+grps)*K, 0:i+grps, :, :] = 1
        else:
            K = iC // oC
            for i in range(0, oC, grps):
                self.mask[i:i+grps, 0:(i+grps)*K, :, :] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super(Conv2dChIncremental, self).forward(x)


# Bu satirin altindaki classlar test edilmedi !!!


class MaskedConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ('A', 'B')
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConvTranspose2d, self).forward(x)


class MaskedConvTranspose2d_zhat_y(nn.ConvTranspose2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ('A')
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0
        in_ch, out_ch, stride_ks = 3, 128, 1
        for name, value in kwargs.items():
            if name == 'stride':
                stride_ks = value
            elif name == 'in_channels':
                in_ch = value
            elif name == 'out_channels':
                out_ch = value
        self.convTranspose2d_blkbyblk = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, stride=stride_ks, kernel_size=stride_ks)

    def forward(self, zhat, y):
        self.weight.data *= self.mask
        zhat_result = super(MaskedConvTranspose2d_zhat_y, self).forward(zhat)
        y_result = self.convTranspose2d_blkbyblk.forward(y)
        return zhat_result + y_result
