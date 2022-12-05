import torch
import torch.nn as nn
import torch.nn.functional as F
import math

lifting_coeff = [-1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 0.869864451624781, 1.149604398860241] # bior4.4
class P_block_v2(nn.Module):
    def __init__(self, linearity_flag=1, csize= 1, conv_filter_size=3, depth_scale=16):
        super(P_block_v2, self).__init__()
        self.conv_filter_size = conv_filter_size
        self.padding = self.conv_filter_size//2
        self.csize= csize


        self.conv1 = nn.Conv2d(1*self.csize,depth_scale*self.csize,self.conv_filter_size,stride=1, padding= self.padding)
        # torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2 * 1 ))
        # torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        # self.conv1.weight.data.fill_(0.01)
        # self.conv1.bias.data.fill_(0.0001)

        self.conv2 = nn.Conv2d(depth_scale*self.csize,depth_scale*self.csize,self.conv_filter_size,stride=1,padding= self.padding)
        # torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2 * 1 ))
        # torch.nn.init.constant_(self.conv2.bias.data, 0.001)
        # self.conv2.weight.data.fill_(0.01)
        # self.conv2.bias.data.fill_(0.0001)

        self.conv3 = nn.Conv2d(depth_scale*self.csize,depth_scale*self.csize,self.conv_filter_size,stride=1,padding= self.padding)
        # torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2 * 1 ))
        # torch.nn.init.constant_(self.conv3.bias.data, 0.001)
        # self.conv3.weight.data.fill_(0.01)
        # self.conv3.bias.data.fill_(0.0001)

        self.conv4 = nn.Conv2d(depth_scale*self.csize,1*self.csize,self.conv_filter_size,stride=1,padding= self.padding)
        # torch.nn.init.xavier_normal_(self.conv4.weight.data, math.sqrt(2 * 1 ))
        # torch.nn.init.constant_(self.conv4.bias.data, 0)
        # self.conv4.weight.data.fill_(0.01)
        # self.conv4.bias.data.fill_(0.01)
        self.linearityFlag = linearity_flag
        self.nonLinearityFunction = nn.Tanh()
    def forward(self,tmp):
        out_res = self.conv1(tmp)
        if (self.linearityFlag==1):
            tmp = self.nonLinearityFunction(out_res)
        else:
            tmp = out_res

        tmp = self.conv2(tmp)
        if (self.linearityFlag==1):
            tmp = self.nonLinearityFunction(tmp)

        tmp = self.conv3(tmp)

        tmp = tmp + out_res
        tmp = self.conv4(tmp)
        return tmp

