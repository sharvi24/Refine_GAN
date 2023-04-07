import torch
import torch.nn as nn
import torch.nn.functional as F

upsample_kernel_sizes = [16, 16, 4, 4]
upsample_rates = [8, 8, 2, 2]
upsample_initial_channel = 512
resblock = "1"
resblock_kernel_sizes = [3, 7, 11]
resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1)),
            nn.utils.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1)),
            nn.utils.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1,))
        ])

        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1)),
            nn.utils.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1)),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1))
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            nn.utils.remove_weight_norm(l)
        for l in self.convs2:
            nn.utils.remove_weight_norm(l)


# class ResBlock2(torch.nn.Module):
#     def __init__(self, channels, kernel_size=3):
#         super(ResBlock2, self).__init__()
#         self.convs = nn.ModuleList([
#             nn.utils.weight_norm(
#                 nn.Conv1d(channels, channels, kernel_size, 1)),
#             nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1))
#         ])

#     def forward(self, x):
#         for c in self.convs:
#             xt = F.leaky_relu(x, 0.1)
#             xt = c(xt)
#             x = xt + x
#         return x

#     def remove_weight_norm(self):
#         for l in self.convs:
#             nn.utils.remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(129, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1  # if resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.utils.weight_norm(
                nn.ConvTranspose1d(
                    upsample_initial_channel//(2**i),
                    upsample_initial_channel//(2**(i+1)),
                    k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k))

        self.conv_post = nn.utils.weight_norm(
            nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            print('inside gen:', x.shape)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            nn.utils.remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_pre)
        nn.utils.remove_weight_norm(self.conv_post)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn1(self.conv2(x)))
        x = self.leaky_relu(self.bn2(self.conv3(x)))
        x = self.leaky_relu(self.bn3(self.conv4(x)))
        x = self.conv5(x)
        return x
