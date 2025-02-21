import torch
import torch.nn as nn


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (
            sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i))
        )


class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class ConditionalPositionalEncoding(nn.Module):
    def __init__(self, channels):
        super(ConditionalPositionalEncoding, self).__init__()
        self.conditional_positional_encoding = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )

    def forward(self, x):
        x = self.conditional_positional_encoding(x)
        return x


class MLP(nn.Module):
    def __init__(self, channels, shallow=True):
        super(MLP, self).__init__()
        expansion = 4
        self.mlp_layer_0 = nn.Conv2d(
            channels, channels * expansion, kernel_size=1, bias=False
        )
        if shallow == True:
            self.mlp_act = nn.GELU()
        else:
            self.mlp_act = Swish()
        self.mlp_layer_1 = nn.Conv2d(
            channels * expansion, channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.mlp_layer_0(x)
        x = self.mlp_act(x)
        x = self.mlp_layer_1(x)
        return x


class LocalAgg(nn.Module):
    def __init__(self, channels):
        super(LocalAgg, self).__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.pointwise_conv_0 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv2d(
            channels, channels, padding=1, kernel_size=3, groups=channels, bias=False
        )
        self.pointwise_prenorm_1 = nn.BatchNorm2d(channels)
        self.pointwise_conv_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_prenorm_1(x)
        x = self.pointwise_conv_1(x)
        return x


class GlobalSparseAttention(nn.Module):
    def __init__(self, channels, r, heads):
        super(GlobalSparseAttention, self).__init__()
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5

        self.num_heads = heads

        self.sparse_sampler = nn.AvgPool2d(kernel_size=1, stride=r)

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = (
            self.qkv(x)
            .view(B, self.num_heads, -1, H * W)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        return x


class LocalPropagation(nn.Module):
    def __init__(self, channels, r):
        super(LocalPropagation, self).__init__()

        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.local_prop = nn.ConvTranspose2d(
            channels, channels, kernel_size=r, stride=r, groups=channels
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, stride=r, bias=False)

    def forward(self, x):
        x = self.local_prop(x)
        x = self.norm(x)
        x = self.proj(x)
        return x


class LGL(nn.Module):
    def __init__(self, channels, r, heads, shallow=True):
        super(LGL, self).__init__()

        self.cpe1 = ConditionalPositionalEncoding(channels)
        self.LocalAgg = LocalAgg(channels)
        self.mlp2 = MLP(channels, shallow)

    def forward(self, x):
        x = self.cpe1(x) + x
        x = self.LocalAgg(x) + x
        x = self.mlp2(x) + x

        return x


class DownSampleLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(DownSampleLayer, self).__init__()
        self.downsample = nn.Conv2d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.downsample(x)
        x = self.norm(x)
        return x


class UpSampleLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(UpSampleLayer, self).__init__()
        self.downsample = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.downsample(x)
        x = self.norm(x)
        return x


class Bas(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        embed_dim=96,
        embeding_dim=228,
        channels=(24, 48, 60),
        conv_r=(4, 3, 2, 2),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        distillation=False,
        dropout=0.3,
    ):
        super(Bas, self).__init__()
        self.Encoder = Encoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            embeding_dim=embeding_dim,
            channels=channels,
            conv_r=conv_r,
            blocks=blocks,
            heads=heads,
            r=r,
            distillation=distillation,
            dropout=dropout,
        )
        self.Decoder = Decoder(
            out_channels=out_channels,
            embed_dim=embed_dim,
            channels=channels,
            conv_r=conv_r,
            blocks=blocks,
            heads=heads,
            r=r,
            distillation=distillation,
            dropout=dropout,
        )

    def forward(self, x):
        embeding, hidden_states_out = self.Encoder(x)
        x = self.Decoder(embeding, hidden_states_out)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=4,
        embed_dim=384,
        embeding_dim=228,
        channels=(48, 96, 240),
        conv_r=(4, 3, 2, 2),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 8),
        r=(4, 2, 2, 1),
        distillation=False,
        dropout=0.3,
    ):
        super(Encoder, self).__init__()
        self.distillation = distillation
        self.conv1 = DownSampleLayer(
            dim_in=in_channels, dim_out=channels[0], r=conv_r[0]
        )
        self.conv2 = DownSampleLayer(
            dim_in=channels[0], dim_out=channels[1], r=conv_r[1]
        )
        self.conv3 = DownSampleLayer(dim_in=channels[1], dim_out=embed_dim, r=conv_r[2])
        bl = []
        for _ in range(blocks[1]):
            bl.append(LGL(channels=channels[1], r=r[1], heads=heads[1], shallow=True))
        self.block2 = nn.Sequential(*bl)
        bl = []
        for _ in range(blocks[2]):
            bl.append(LGL(channels=embed_dim, r=r[2], heads=heads[2], shallow=False))
        self.block3 = nn.Sequential(*bl)
        self.position_embeddings = nn.Parameter(torch.zeros(1, embeding_dim, embed_dim))
        self.pooling1 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden_states_out = []
        x = self.conv1(x)
        x = self.pooling1(x)
        hidden_states_out.append(x)
        x = self.conv2(x)
        x = self.block2(x)
        hidden_states_out.append(x)
        x = self.conv3(x)
        x = self.block3(x)
        B, C, W, H = x.shape
        x = x.flatten(2).transpose(-1, -2)
        x = x + self.position_embeddings
        x = self.dropout(x)
        x = x.reshape(B, C, W, H)
        return x, hidden_states_out


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=3,
        embed_dim=384,
        channels=(48, 96, 240),
        conv_r=(4, 3, 2, 2),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 8),
        r=(4, 2, 2, 1),
        distillation=False,
        dropout=0.3,
    ):
        super(Decoder, self).__init__()
        self.distillation = distillation

        self.conv1 = UpSampleLayer(
            dim_in=channels[0], dim_out=out_channels, r=conv_r[0]
        )
        self.conv2 = UpSampleLayer(dim_in=channels[1], dim_out=channels[0], r=conv_r[1])
        self.conv3 = UpSampleLayer(dim_in=embed_dim, dim_out=channels[1], r=conv_r[2])

        bl = []
        for _ in range(blocks[0]):
            bl.append(LGL(channels=channels[0], r=r[0], heads=heads[0], shallow=True))
        self.block1 = nn.Sequential(*bl)
        bl = []
        for _ in range(blocks[1]):
            bl.append(LGL(channels=channels[1], r=r[1], heads=heads[1], shallow=False))
        self.block2 = nn.Sequential(*bl)

    def forward(self, x, hidden_states_out):
        x = self.conv3(x)
        x = x + hidden_states_out[1]
        x = self.block2(x)
        x = self.conv2(x)
        x = x + hidden_states_out[0]
        x = self.block1(x)
        x = self.conv1(x)
        return x
