import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding, down):
        super(ResidualConv, self).__init__()
        self.down = down
        if down == 'down':
            self.updown = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.ReLU(),
                nn.Conv2d(
                    input_dim, input_dim, kernel_size=3, stride=stride, padding=padding
                ),
            )
        elif down == 'up':
            self.updown = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    input_dim, output_dim, kernel_size=2, stride=stride, padding=padding
                ),
            )
        elif down == 'flat':
            self.updown = Squeeze_Excite_Block(output_dim, 2)
        self.conv_block = nn.Sequential(
            # nn.BatchNorm2d(input_dim),
            # nn.ReLU(),
            # nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=3, dilation=3),
            nn.Dropout2d(0.25)
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x, y=None):
        if self.down == 'down':
            updown = self.updown(x)
            out = self.conv_block(updown) + self.conv_skip(updown)
        elif self.down == 'up':
            updown = torch.cat([self.updown(x), y], dim=1)
            out = self.conv_block(updown) + self.conv_skip(updown)
        elif self.down == 'flat':
            updown = torch.cat([self.updown(x), y], dim=1)
            out = self.conv_block(updown) + self.conv_skip(updown)
        return out


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# without BN version
class ASPP2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ASPP2, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[1, 2, 3]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, scale=2):
        super(Upsample_, self).__init__()
        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale, align_corners=False)
        self.residual_conv = ResidualConv(input_dim, output_dim, stride, 1)

    def forward(self, x):
        return self.residual_conv(self.upsample(x))


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            # nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3),
        )
        self.conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        conv = self.conv(torch.cat((branch1, branch2, branch3, branch4), dim=1))
        return conv


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * input


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False, stride=2)  # 7,3     3,1
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = self.conv2(self.relu1(self.conv1(avg_out)))
        max_out = self.conv2(self.relu1(self.conv1(max_out)))
        # x = torch.cat([avg_out, max_out], dim=1)
        # x = self.conv1(x)
        out = avg_out + max_out
        return self.sigmoid(out) * input


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        ca = self.ca(x)
        sa = self.sa(ca)
        return sa


class MFISnet(nn.Module):
    def __init__(self, channel, filters=None):
        super(MFISnet, self).__init__()

        if filters is None:
            filters = [16, 32, 64, 128, 256, 512, 1024]

        self.down1_1 = ASPP(channel, filters[0])
        self.ca1 = Squeeze_Excite_Block(filters[0], 2)
        self.down2_1 = ResidualConv(filters[0], filters[1], 2, 1, 'down')
        self.ca2 = Squeeze_Excite_Block(filters[1], 2)
        self.down3_1 = ResidualConv(filters[1], filters[2], 2, 1, 'down')
        self.ca3 = Squeeze_Excite_Block(filters[2], 2)
        self.down4 = ResidualConv(filters[2], filters[3], 2, 1, 'down')
        self.ca4 = Squeeze_Excite_Block(filters[3], 2)
        self.down5 = ResidualConv(filters[3], filters[4], 2, 1, 'down')
        self.ca5 = Squeeze_Excite_Block(filters[4], 2)
        self.bridge = ASPP(filters[4], filters[4])
        self.ca6 = Squeeze_Excite_Block(filters[4], 2)
        self.up4 = ResidualConv(filters[4], filters[3], 2, 0, 'up')
        self.ca7 = Squeeze_Excite_Block(filters[3], 2)
        self.up3_1 = ResidualConv(filters[3], filters[2], 2, 0, 'up')
        self.ca8 = Squeeze_Excite_Block(filters[2], 2)
        self.up2_1 = ResidualConv(filters[2], filters[1], 2, 0, 'up')
        self.ca9 = Squeeze_Excite_Block(filters[1], 2)
        self.up1_1 = ResidualConv(filters[1], filters[0], 2, 0, 'up')

        self.down3_2 = ResidualConv(filters[3], filters[2], 2, 0, 'up')
        self.down2_2 = ResidualConv(filters[2], filters[1], 2, 0, 'up')
        self.down2_3 = ResidualConv(filters[2], filters[1], 2, 0, 'up')
        self.down1_2 = ResidualConv(filters[1], filters[0], 2, 0, 'up')
        self.down1_3 = ResidualConv(filters[1], filters[0], 2, 0, 'up')
        self.down1_4 = ResidualConv(filters[1], filters[0], 2, 0, 'up')

        self.up3_2 = ResidualConv(filters[3], filters[2], 2, 0, 'up')
        self.cat3_2 = ResidualConv(filters[3], filters[2], 1, 1, 'flat')
        self.up2_2 = ResidualConv(filters[2], filters[1], 2, 0, 'up')
        self.cat2_2 = ResidualConv(filters[2], filters[1], 1, 1, 'flat')
        self.up2_3 = ResidualConv(filters[2], filters[1], 2, 0, 'up')
        self.cat2_3 = ResidualConv(filters[2], filters[1], 1, 1, 'flat')
        self.up1_2 = ResidualConv(filters[1], filters[0], 2, 0, 'up')
        self.cat1_2 = ResidualConv(filters[1], filters[0], 1, 1, 'flat')
        self.up1_3 = ResidualConv(filters[1], filters[0], 2, 0, 'up')
        self.cat1_3 = ResidualConv(filters[1], filters[0], 1, 1, 'flat')
        self.up1_4 = ResidualConv(filters[1], filters[0], 2, 0, 'up')
        self.cat1_4 = ResidualConv(filters[1], filters[0], 1, 1, 'flat')

        self.CBAM1 = CBAM(filters[0], 2, 3)
        self.CBAM2 = CBAM(filters[0], 2, 3)
        self.CBAM3 = CBAM(filters[0], 2, 3)
        self.CBAM4 = CBAM(filters[0], 2, 3)

        self.end = ASPP(filters[0] * 4, filters[0] * 2)
        self.out = nn.Sequential(
            nn.Conv2d(filters[0] * 2, filters[0], 3, 1, 1),
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        down1_1 = self.down1_1(x)  # 1->16
        ca1 = self.ca1(down1_1)  # 16
        down2_1 = self.down2_1(ca1)  # 32
        ca2 = self.ca2(down2_1)  # 32
        down3_1 = self.down3_1(ca2)  # 64
        ca3 = self.ca3(down3_1)  # 64
        down4 = self.down4(ca3)  # 128
        ca4 = self.ca4(down4)  # 128
        down5 = self.down5(ca4)  # 256
        ca5 = self.ca5(down5)
        bridge = self.bridge(ca5)
        ca6 = self.ca6(bridge)
        up4 = self.up4(ca6, down4)
        ca7 = self.ca7(up4)
        up3_1 = self.up3_1(ca7, down3_1)
        ca8 = self.ca8(up3_1)
        up2_1 = self.up2_1(ca8, down2_1)
        ca9 = self.ca9(up2_1)
        up1_1 = self.up1_1(ca9, down1_1)

        down3_2 = self.down3_2(down4, down3_1)
        down2_2 = self.down2_2(down3_1, down2_1)
        down2_3 = self.down2_3(down3_2, down2_2)
        down1_2 = self.down1_2(down2_1, down1_1)
        down1_3 = self.down1_3(down2_2, down1_2)
        down1_4 = self.down1_4(down2_3, down1_3)

        up3_2 = self.up3_2(up4, up3_1)
        up3_2 = self.cat3_2(down3_2, up3_2)
        up2_2 = self.up2_2(up3_1, up2_1)
        up2_2 = self.cat2_2(down2_2, up2_2)
        up2_3 = self.up2_3(up3_2, up2_2)
        up2_3 = self.cat2_3(down2_3, up2_3)
        up1_2 = self.up1_2(up2_1, up1_1)
        up1_2 = self.cat1_2(down1_2, up1_2)
        up1_3 = self.up1_3(up2_2, up1_2)
        up1_3 = self.cat1_3(down1_3, up1_3)
        up1_4 = self.up1_4(up2_3, up1_3)
        up1_4 = self.cat1_4(down1_4, up1_4)

        out1 = self.CBAM1(up1_1)
        out2 = self.CBAM2(up1_2)
        out3 = self.CBAM3(up1_3)
        out4 = self.CBAM4(up1_4)

        end = self.end(torch.cat([out1, out2, out3, out4], dim=1))
        out = self.out(end)
        return out
