import torch
import numpy as np
import torchvision
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm as sn


def get_backbone(exp_dict):
    # nclasses = exp_dict["num_classes"]
    backbone_name = exp_dict["backbone"]["name"].lower()
    if backbone_name == "biggan_encoder":
        backbone = Encoder(exp_dict)
    elif backbone_name == "biggan_decoder":
        backbone = Generator(exp_dict)
    elif backbone_name == "resnet18":
        backbone = ResNet18(exp_dict["dataset"]["num_classes"])
    elif backbone_name == "mlp":
        ni = exp_dict["dataset"]["n_attributes"]
        no = exp_dict["dataset"]["num_classes"]
        nhidden = exp_dict["backbone"]["n_hidden"]
        nlayers = exp_dict["backbone"]["num_layers"]
        backbone = MLP(ni, no, nhidden, nlayers)
    else:
        raise ValueError

    return backbone


def get_resnet_output_size(ratio, width):
    if ratio == 32 or ratio == 16:
        output_size = 128 * width
    elif ratio == 8:
        output_size = 64 * width
    else:
        raise ValueError("Incorrect Ratio")

    return output_size


class ResnetBlock(torch.nn.Module):
    def __init__(self, ni, no, stride, activation, spectral_norm=False, dp_prob=.3):
        super().__init__()
        self.activation = activation
        self.bn0 = torch.nn.BatchNorm2d(ni)
        self.conv0 = torch.nn.Conv2d(ni, no, 3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(no)
        self.conv1 = torch.nn.Conv2d(no, no, 3, stride=1, padding=1, bias=False)
        self.dropout1 = torch.nn.Dropout2d(dp_prob)
        if stride > 1:
            self.downsample = True
        else:
            self.downsample = False
        if ni != no:
            self.reduce = True
            self.conv_reduce = torch.nn.Conv2d(ni, no, 1, stride=1, padding=0, bias=False)
        else:
            self.reduce = False
        if spectral_norm:
            self.conv0 = sn(self.conv0)
            self.conv1 = sn(self.conv1)
            if self.reduce:
                self.conv_reduce = sn(self.conv_reduce)

    def forward(self, x):
        y = self.activation(self.bn0(x))
        y = self.conv0(y)
        y = self.activation(self.bn1(y))
        y = self.dropout1(y)
        y = self.conv1(y)
        if self.reduce:
            x = self.conv_reduce(x)
        if self.downsample:
            y = F.avg_pool2d(y, 2, 2)
            x = F.avg_pool2d(x, 2, 2)
        return x + y


class ResnetGroup(torch.nn.Module):
    def __init__(self, n, ni, no, stride, activation, spectral_norm, dp_prob=.3):
        super().__init__()
        self.n = n
        in_plane = ni
        for i in range(n):
            setattr(self, "block_%d" %i, ResnetBlock(in_plane, no, stride if i==0 else 1, activation, spectral_norm, dp_prob=dp_prob))
            in_plane = no
    def forward(self, x):
        for i in range(self.n):
            x = getattr(self, "block_%d" %i)(x)
        return x


class Resnet(torch.nn.Module):
    def __init__(self, ratio=0, width=1, activation=F.relu, spectral_norm=False, dp_prob=.3):
        super().__init__()
        self.width = width

        self.width = width
        if ratio == 32:
            self.channels = (np.array([16, 32, 64, 128, 128]) * width).astype(int)
            self.nblocks = [2, 1, 1, 1, 3]
            in_resolution = 128
        elif ratio == 16:
            self.channels = (np.array([16, 32, 64, 128]) * width).astype(int)
            self.nblocks = [1, 1, 1, 1]
            in_resolution = 64
        elif ratio == 8:
            self.channels = (np.array([32, 32, 64]) * width).astype(int)
            self.nblocks = [2, 1, 2]
            in_resolution = 32
        else:
            raise ValueError

        self.output_size = self.channels[-1]
        in_ch = 3
        for i, (out_ch, nblocks) in enumerate(zip(self.channels, self.nblocks)):
            setattr(self, "group%d" %i, ResnetGroup(nblocks, in_ch, out_ch, stride=2, activation=activation, spectral_norm=spectral_norm, dp_prob=dp_prob))
            in_resolution = in_resolution // 2
            assert(in_resolution > 2)
            in_ch = out_ch

    def forward(self, x):
        b = x.size(0)
        for i in range(len(self.channels)):
            x = getattr(self, "group%d" %i)(x)
        return x


class MultiHeadMLP(torch.nn.Module):
    def __init__(self, ni, heads_no, nhidden, nlayers):
        super().__init__()
        self.nlayers = nlayers
        self.heads_no = heads_no

        for i in range(nlayers):
            if i == 0:
                setattr(self, "linear%d" %i, torch.nn.Linear(ni, nhidden, bias=False))
            else:
                setattr(self, "linear%d" %i, torch.nn.Linear(nhidden, nhidden, bias=False))
            setattr(self, "bn%d" %i, torch.nn.BatchNorm1d(nhidden))
        if nlayers == 0:
            nhidden = ni

        for i, no in enumerate(heads_no):
            setattr(self, "linear_out_%d" %i, torch.nn.Linear(nhidden, no))

    def forward(self, x):
        for i in range(self.nlayers):
            linear = getattr(self, "linear%d" %i)
            bn = getattr(self, "bn%d" %i)
            x = linear(x)
            x = F.leaky_relu(bn(x), 0.2, True)

        return [getattr(self, "linear_out_%d" %i)(x) for i in range(len(self.heads_no))]


class MLP(torch.nn.Module):
    def __init__(self, ni, no, nhidden, nlayers):
        super().__init__()
        self.nlayers = nlayers

        for i in range(nlayers):
            if i == 0:
                setattr(self, "linear%d" %i, torch.nn.Linear(ni, nhidden, bias=False))
            else:
                setattr(self, "linear%d" %i, torch.nn.Linear(nhidden, nhidden, bias=False))
            setattr(self, "bn%d" %i, torch.nn.BatchNorm1d(nhidden))
        if nlayers == 0:
            nhidden = ni

        self.linear_out = torch.nn.Linear(nhidden, no)

    def forward(self, x):
        for i in range(self.nlayers):
            linear = getattr(self, "linear%d" %i)
            bn = getattr(self, "bn%d" %i)
            x = linear(x)
            x = F.leaky_relu(bn(x), 0.2, True)

        return self.linear_out(x)


class EncoderBackbone(Resnet):
    def __init__(self, mlp, ratio=0, width=1, dp_prob=.3, pooling_last=True, return_features=True):
        super().__init__(ratio=ratio, width=width, activation=torch.nn.ReLU(True), dp_prob=dp_prob)
        self.bn_out = torch.nn.BatchNorm2d(self.channels[-1])
        self.pooling_last = pooling_last
        self.return_features = return_features

        self.mlp = mlp

    def forward(self, x):
        for i in range(len(self.channels)):
            x = getattr(self, "group%d" %i)(x)
        x = F.leaky_relu(self.bn_out(x), 0.2, inplace=True)

        out = x
        if self.pooling_last:
            out = x.mean(3).mean(2).view(x.size(0), -1)

        if self.return_features:
            return x, self.mlp(x.mean(3).mean(2).view(x.size(0), -1))

        return self.mlp(out.view(x.size(0), -1))


class Discriminator(Resnet):
    def __init__(self, *args, n_out=1, **kwargs):
        super().__init__(*args, activation=torch.nn.LeakyReLU(0.2, inplace=True), spectral_norm=True, **kwargs)
        self.classifier = torch.nn.Linear(self.channels[-1], n_out)
        self.bn_out = torch.nn.BatchNorm2d(self.channels[-1])

    def forward(self, x, mask=[0,0,0,0]):
        b = x.size(0)
        features = None
        for i in range(len(self.channels)):
            x = getattr(self, "group%d" %i)(x)
            if mask[i]:
                if features is None:
                    features = x.clone().view(b, -1)
                else:
                    features = torch.cat((features, x.clone().view(b, -1)), 1)

        F.relu(self.bn_out(x), True)
        x = x.mean(3).mean(2)
        return features, self.classifier(x)


class DiscriminatorLoss(torch.nn.Module):
    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, x, real=False):
        labels = torch.ones(x.size(0), dtype=torch.float, device=x.device)
        if real:
            labels = labels * 0 + 0.1
        else:
            labels -= 0.1
        _, logits = self.discriminator(x)
        return F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1))


class InterpolateResidualGroup(torch.nn.Module):
    def __init__(self, nblocks, ni, no, z_dim, upsample=False):
        super().__init__()
        self.nblocks = nblocks
        for n in range(nblocks):
            if n == 0:
                setattr(self, "block%d" %n, InterpolateResidualBlock(ni, no, z_dim, upsample=upsample))
            else:
                setattr(self, "block%d" %n, InterpolateResidualBlock(no, no, z_dim, upsample=False))

    def forward(self, x, z):
        for n in range(self.nblocks):
            block = getattr(self, "block%d" %n)
            x = block(x, z)
        return x


class ConditionalBatchNorm(torch.nn.Module):
    def __init__(self, no, z_dim):
        super().__init__()
        self.no = no
        self.bn = torch.nn.BatchNorm2d(no, affine=False)
        self.condition = torch.nn.Linear(z_dim, 2 * no)

    def forward(self, x, z):
        cond = self.condition(z).view(-1, 2 * self.no, 1, 1)
        return self.bn(x) * cond[:, :self.no] + cond[:, self.no:]


class InterpolateResidualBlock(torch.nn.Module):
    def __init__(self, ni, no, z_dim, upsample=False):
        super().__init__()
        self.bn0 = ConditionalBatchNorm(ni, z_dim)
        self.conv0 = torch.nn.Conv2d(ni, no, 3, 1, 1, bias=False)
        self.conv0 = sn(self.conv0)
        self.bn1 = ConditionalBatchNorm(no, z_dim)
        self.conv1 = torch.nn.Conv2d(no, no, 3, 1, 1, bias=False)
        self.conv1 = sn(self.conv1)
        self.upsample = upsample
        self.reduce = ni != no
        if self.reduce:
            self.conv_short = sn(torch.nn.Conv2d(ni, no, 1, 1, 0, bias=False))

    def forward(self, x, z):
        if self.upsample:
            shortcut = F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            shortcut = x
        x = F.relu(self.bn0(x, z), True)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv0(x)
        x = F.relu(self.bn1(x, z), True)
        x = self.conv1(x)
        if self.reduce:
            x = self.conv_short(shortcut) + x
        else:
            x = x + shortcut
        return x


class Decoder(torch.nn.Module):
    def __init__(self, z_dim, width, in_ch, ratio, in_h, in_w, mlp_width, mlp_depth):
        super().__init__()
        self.mlp = MLP(z_dim, in_ch * in_h * in_w, in_ch * mlp_width, mlp_depth)
        self.in_ch = in_ch
        self.in_h = in_h
        self.in_w = in_w
        self.ratio = ratio
        self.width = width
        self.channels = []
        self.z_dim = z_dim
        if ratio == 32:
            self.channels = (np.array([128, 64, 32, 16, 16]) * width).astype(int)
            self.nblocks = [1, 1, 2, 2, 1]
            in_resolution = 128
        elif ratio == 16:
            self.channels = (np.array([128, 64, 32, 16]) * width).astype(int)
            self.nblocks = [1, 1, 1, 1]
            in_resolution = 64
        elif ratio == 8:
            self.channels = (np.array([64, 32, 16]) * width).astype(int)
            self.nblocks = [1, 1, 1]
            in_resolution = 32
        else:
            raise ValueError

        for i, out_ch in enumerate(self.channels):
            setattr(self, 'group%d'%i, InterpolateResidualGroup(self.nblocks[i], in_ch, out_ch, z_dim, upsample=True))
            in_ch = out_ch
        self.bn_out = torch.nn.BatchNorm2d(self.channels[-1])
        self.conv_out = torch.nn.Conv2d(self.channels[-1], 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):

        z = z.view(z.size(0), -1)
        x = self.mlp(z)
        x = x.view(-1, self.in_ch, self.in_h, self.in_w)
        for i in range(len(self.channels)):
            group = getattr(self, "group%d" %i)
            x = group(x, z)
        x = F.relu(self.bn_out(x), True)
        return torch.tanh(self.conv_out(x))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        try:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        except:
            pass


class Encoder(torch.nn.Module):

    def __init__(self, exp_dict):
        super().__init__()
        self.exp_dict=exp_dict
        self.w = self.exp_dict["dataset"]["width"]
        self.h = self.exp_dict["dataset"]["height"]

        if min(self.w, self.h) == 128:
            self.ratio = 32
        elif min(self.w, self.h) == 64:
            self.ratio = 16
        elif min(self.w, self.h) == 32:
            self.ratio = 8

        self.channels_width = exp_dict["backbone"]["channels_width"]
        mlp_width = self.exp_dict["backbone"]["mlp_width"]
        mlp_depth = self.exp_dict["backbone"]["mlp_depth"]
        n_continuous = exp_dict["dataset"]["n_continuous"]
        mlp_heads = [n_continuous, 48, 48]

        if exp_dict["backbone"]["feature_extractor"].lower() == "resnet":
            self.stem_feature_size = get_resnet_output_size(self.ratio, self.channels_width)
            mlp = MultiHeadMLP(self.stem_feature_size, mlp_heads, mlp_width * self.stem_feature_size, mlp_depth)
            self.encoder = EncoderBackbone(mlp,
                                   ratio=self.ratio,
                                   width=self.channels_width,
                                   dp_prob=exp_dict["backbone"]["dp_prob"],
                                   return_features=True,
                                   pooling_last=True)


        self.weights_init = weights_init

        self.apply(self.weights_init)


    def encode(self, x):

        z, predictions = self.encoder(x)

        pred_continuous, pred_font, pred_char = predictions

        return dict(z=z, pred_continuous=pred_continuous,
                    pred_font=pred_font, pred_char=pred_char)


    def forward(self, x):

        return self.encode(x)


class Generator(torch.nn.Module):

    def __init__(self, exp_dict):

        super().__init__()
        self.exp_dict = exp_dict
        self.z_dim = exp_dict["z_dim"]
        self.w = self.exp_dict["dataset"]["width"]
        self.h = self.exp_dict["dataset"]["height"]

        if min(self.w, self.h) == 128:
            self.ratio = 32
        elif min(self.w, self.h) == 64:
            self.ratio = 16
        elif min(self.w, self.h) == 32:
            self.ratio = 8

        self.output_w = self.w // self.ratio
        self.output_h = self.h // self.ratio
        self.channels_width = exp_dict["backbone"]["channels_width"]
        mlp_width = self.exp_dict["backbone"]["mlp_width"]
        mlp_depth = self.exp_dict["backbone"]["mlp_depth"]
        n_continuous = exp_dict["dataset"]["n_continuous"]

        self.char_embedding = torch.nn.Embedding(48, self.z_dim)
        self.font_embedding = torch.nn.Embedding(48, self.z_dim)

        self.decoder = Decoder(n_continuous + self.z_dim * 2,
                                self.channels_width,
                                in_ch=1,
                                ratio=self.ratio,
                                in_h=self.output_h,
                                in_w=self.output_w,
                                mlp_width=mlp_width,
                                mlp_depth=mlp_depth)


    def embed_attributes(self, categorical, continuous):

        inputs = []

        inputs.append(self.char_embedding(categorical[:, 0]))
        inputs.append(self.font_embedding(categorical[:, 1]))
        inputs.append(continuous)

        return torch.cat((inputs), 1)


    def decode(self, inputs):
        return self.decoder(inputs)


    def forward(self, categorical, continuous):
        inputs = self.embed_attributes(categorical, continuous)

        reconstruction = self.decode(inputs)

        return reconstruction


class ResNet18(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feat_extract = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.feat_extract(x)
