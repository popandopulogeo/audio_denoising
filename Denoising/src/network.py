
import torch
import torch.nn as nn
import torch.nn.functional as F

# Networks
##############################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size:tuple, dilation:tuple,
                 stride=1,
                 norm_fn='bn',
                 act='relu'):
        super(ConvBlock, self).__init__()
        pad = ((kernel_size[0] - 1) // 2 * dilation[0], (kernel_size[1] - 1) // 2 * dilation[1])
        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation, bias=norm_fn is None))
        if norm_fn == 'bn':
            block.append(nn.BatchNorm2d(out_channels))
        if act == 'relu':
            block.append(nn.ReLU())
        elif act == 'prelu':
            block.append(nn.PReLU())
        elif act == 'lrelu':
            block.append(nn.LeakyReLU())
        elif act == 'tanh':
            block.append(nn.Tanh())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1,
                 norm_fn='bn',
                 act='prelu'):
        super(DownConvBlock, self).__init__()
        pad = (kernel_size - 1) // 2 * dilation
        block = []
        block.append(nn.ReflectionPad2d(pad))
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, bias=norm_fn is None))
        if norm_fn == 'bn':
            block.append(nn.BatchNorm2d(out_channels))
        if act == 'prelu':
            block.append(nn.PReLU())
        elif act == 'lrelu':
            block.append(nn.LeakyReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1,
                 norm_fn='bn',
                 act='prelu',
                 up_mode='upconv'):
        super(UpConvBlock, self).__init__()
        pad = (kernel_size - 1) // 2 * dilation
        block = []
        if up_mode == 'upconv':
            block.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, pad, dilation,
                                            bias=norm_fn is None))
        elif up_mode == 'upsample':
            block.append(nn.Upsample(scale_factor=2))
            block.append(nn.ReflectionPad2d(pad))
            block.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, 0, dilation,
                                   bias=norm_fn is None))
        if norm_fn == 'bn':
            block.append(nn.BatchNorm2d(out_channels))
        if act == 'prelu':
            block.append(nn.PReLU())
        elif act == 'lrelu':
            block.append(nn.LeakyReLU())
        elif act == 'tanh':
            block.append(nn.Tanh())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class NoiseRemovalNet(nn.Module):
    def __init__(self, freq_bins=256, nf=96):
        super(NoiseRemovalNet, self).__init__()
        kernel_sizes = [(1, 7), (7, 1), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5)]
        dilations = [(1, 1), (1, 1), (1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
        self.encoder_x = self.make_encoder(kernel_sizes, dilations, nf)
        self.encoder_n = self.make_encoder(kernel_sizes, dilations, nf // 2, outf=4)

        self.lstm = nn.LSTM(input_size=8*freq_bins + 4*freq_bins, hidden_size=200, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(400, 600),
                                nn.ReLU(True),
                                nn.Linear(600, 600),
                                nn.ReLU(True),
                                nn.Linear(600, freq_bins * 2),
                                nn.Sigmoid())

    def make_encoder(self, kernel_sizes, dilations, nf=96, outf=8):
        encoder_x = []
        for i in range(len(kernel_sizes)):
            if i == 0:
                encoder_x.append(ConvBlock(2, nf, kernel_sizes[i], dilations[i]))
            else:
                encoder_x.append(ConvBlock(nf, nf, kernel_sizes[i], dilations[i]))
        encoder_x.append(ConvBlock(nf, outf, (1, 1), (1, 1)))
        return nn.Sequential(*encoder_x)

    def forward(self, x, n):
        f_x = self.encoder_x(x)
        f_x = f_x.view(f_x.size(0), -1, f_x.size(3)).permute(2, 0, 1)
        f_n = self.encoder_n(n)
        f_n = f_n.view(f_n.size(0), -1, f_n.size(3)).permute(2, 0, 1)
        self.lstm.flatten_parameters()
        f_x, _ = self.lstm(torch.cat([f_x, f_n], dim=2))
        f_x = f_x.permute(1, 0, 2)
        f_x = self.fc(f_x)
        out = f_x.permute(0, 2, 1).view(f_x.size(0), 2, -1, f_x.size(1))
        return out

class NoiseEstimationNet(nn.Module):
    def __init__(self):
        super(NoiseEstimationNet, self).__init__()
        ch1 = 64
        ch2 = 128
        ch3 = 256
        self.down1 = nn.Sequential(
            DownConvBlock(2, ch1, 5, 1),
        )
        self.down2 = nn.Sequential(
            DownConvBlock(ch1, ch2, 5, 2),
            DownConvBlock(ch2, ch2, 5, 1),
        )
        self.down3 = nn.Sequential(
            DownConvBlock(2, ch1, 5, 1),
        )
        self.down4 = nn.Sequential(
            DownConvBlock(ch1, ch2, 5, 2),
            DownConvBlock(ch2, ch2, 5, 1),
        )
        self.mid = nn.Sequential(
            DownConvBlock(ch2 * 2, ch3, 3, 2),
            DownConvBlock(ch3, ch3, 3, 1),
            DownConvBlock(ch3, ch3, 3, 1, dilation=2),
            DownConvBlock(ch3, ch3, 3, 1, dilation=4),
            DownConvBlock(ch3, ch3, 3, 1, dilation=8),
            DownConvBlock(ch3, ch3, 3, 1, dilation=16),
            DownConvBlock(ch3, ch3, 3, 1),
            DownConvBlock(ch3, ch3, 3, 1),
            UpConvBlock(ch3, ch2, 3, 2),
        )
        self.up1 = nn.Sequential(
            DownConvBlock(ch2 * 2, ch2, 3, 1),
            UpConvBlock(ch2, ch1, 3, 2),
        )
        self.up2 = nn.Sequential(
            DownConvBlock(ch1 * 2, ch1, 3, 1),
            DownConvBlock(ch1, 2, 3, 1, norm_fn=None, act=None)
        )

    def forward(self, x, y):
        down1 = self.down1(x)
        down2 = self.down2(down1)

        down3 = self.down3(y)
        down4 = self.down4(down3)
        out = self.mid(torch.cat([down2, down4], dim=1))
        if out.shape != down4.shape:
            out = F.interpolate(out, down4.size()[-2:])
        out = self.up1(torch.cat([out, down4], dim=1))
        if out.shape != down3.shape:
            out = F.interpolate(out, down3.size()[-2:])
        out = self.up2(torch.cat([out, down3], dim=1))
        return out

class SilentIntervalDetection(nn.Module):
    def __init__(self, freq_bins=256, time_bins=203, nf=96):
        super(SilentIntervalDetection, self).__init__()
        kernel_sizes = [(1, 7), (7, 1), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5), (5, 5)]
        dilations    = [(1, 1), (1, 1), (1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (1, 1), (2, 2), (4, 4)]
        self.encoder_audio = self.make_encoder(kernel_sizes, dilations, nf=nf, outf=8)

        self.lstm = nn.LSTM(input_size=8*freq_bins, hidden_size=100, bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear(200, 100),
                                 nn.ReLU(True),
                                 nn.Linear(100, 1))

    def make_encoder(self, kernel_sizes, dilations, nf=96, outf=8):
        encoder_x = []
        for i in range(len(kernel_sizes)):
            if i == 0:
                encoder_x.append(ConvBlock(2, nf, kernel_sizes[i], dilations[i]))
            else:
                encoder_x.append(ConvBlock(nf, nf, kernel_sizes[i], dilations[i]))
        encoder_x.append(ConvBlock(nf, outf, (1, 1), (1, 1)))
        return nn.Sequential(*encoder_x)

    def forward(self, s):
        f_s = self.encoder_audio(s)
        f_s = f_s.view(f_s.size(0), -1, f_s.size(3)) # (B, C1, T1)
        # f_s = F.interpolate(f_s, size=v_num_frames) # (B, C2, T1)
        merge = f_s.permute(2, 0, 1)  # (T1, B, C1+C2)
        self.lstm.flatten_parameters()
        merge, _ = self.lstm(merge)
        merge = merge.permute(1, 0, 2)# (B, T1, C1+C2)
        merge = self.fc1(merge)
        out = merge.squeeze(2)
        out = nn.Sigmoid()(out)
        return out
        
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.silent_interval_detection = SilentIntervalDetection()
        self.noise_estimation = NoiseEstimationNet()
        self.noise_removal = NoiseRemovalNet()

    def forward(self, x):
        sid_mask = self.silent_interval_detection(x)
        sid_mask = sid_mask.view(sid_mask.shape[0], 1, 1, sid_mask.shape[1])
        noise_intervals = torch.round(sid_mask) * x
        noise_pred = self.noise_estimation(x, noise_intervals)
        noise_mask = self.noise_removal(x, noise_pred)
        return sid_mask.view(sid_mask.shape[0], -1), noise_mask, noise_pred