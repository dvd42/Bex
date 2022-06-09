import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from haven import haven_utils as hu
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from .backbones import get_backbone, Discriminator, DiscriminatorLoss
# from .backbones import Discriminator
# from .backbones import get_backbone
# from .backbones import DiscriminatorLoss


def l1_loss(x, reconstruction):
    #Safe version of l1_loss
    pix_mse = F.l1_loss(x, reconstruction, reduction="mean")
    pix_mse = pix_mse * (pix_mse != 0) #/ x.size(0) # masking to avoid nan
    return pix_mse


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


class Generator(torch.nn.Module):

    def __init__(self, exp_dict, writer=None):
        """ Constructor
        Args:
            model: architecture to train
            self.exp_dict: reference to dictionary with the global state of the application
        """
        super().__init__()
        if "generator_dict" in exp_dict:
            self.exp_dict = exp_dict["generator_dict"]
        else:
            self.exp_dict = exp_dict
        self.model = get_backbone(self.exp_dict)
        self.ngpu = self.exp_dict["ngpu"]
        self.devices = list(range(self.ngpu))
        # self.is_categorical = {}
        self.z_dim = self.exp_dict["z_dim"]
        self.w = self.exp_dict["dataset"]["width"]
        self.h = self.exp_dict["dataset"]["height"]
        self.alpha = self.exp_dict["alpha"]
        self.lamb = self.exp_dict["lambda"]

        if min(self.w, self.h) == 128:
            self.ratio = 32
        elif min(self.w, self.h) == 64:
            self.ratio = 16
        elif min(self.w, self.h) == 32:
            self.ratio = 8

        self.output_w = self.w // self.ratio
        self.output_h = self.h // self.ratio
        self.channels_width = self.exp_dict["backbone"]["channels_width"]

        self.model.cuda()
        self.discriminator = Discriminator(ratio=self.ratio, width=self.channels_width)
        self.discriminator.apply(weights_init)
        self.discriminator.cuda()
        self.discriminator_loss = DiscriminatorLoss(self.discriminator)

        if self.ngpu > 1:
            self.model = torch.nn.DataParallel(self.model, list(range(self.ngpu)))
            self.discriminator = torch.nn.DataParallel(self.discriminator, list(range(self.ngpu)))

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                          betas=(0.9, 0.999),
                                          lr=self.exp_dict["lr"], weight_decay=self.exp_dict["weight_decay"])

        self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(),
                                          betas=(0.9, 0.999),
                                          lr=self.exp_dict["lr"], weight_decay=self.exp_dict["weight_decay"])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.exp_dict["max_epoch"])

        self.oracle = None
        # self.oracle = self.load_oracle().cuda()
        if "weights" in self.exp_dict:
            self.load_state_dict(hu.torch_load(self.exp_dict["weights"]))
        # self.freeze_oracle()

    def freeze_oracle(self):

        # Freeze oracle
        for p in self.oracle.parameters():
            p.requires_grad_(False)

        self.oracle.eval()


    def load_oracle(self):


        oracle_dict = self.exp_dict["encoder_dict"]
        oracle = get_backbone(oracle_dict)
        if "weights" in oracle_dict:
            weights = oracle_dict["weights"]
        else:
            weights = os.path.join(self.exp_dict["savedir_base"], hu.hash_dict(oracle_dict), "model.pth")

        oracle.load_state_dict(hu.torch_load(weights)["model"])

        return oracle


    def save_img(self, name, images, reconstructions, idx=0):

        f, ax = plt.subplots(1, 2)
        f.set_size_inches(15.5, 15.5)
        ax[0].imshow(images.cpu().permute(1, 2, 0))
        ax[1].imshow(reconstructions.cpu().permute(1, 2, 0))
        ax[0].axis('off')
        ax[1].axis('off')

        wandb.log({name: f}, commit=False)
        plt.close()


    def train_on_batch(self, batch_idx, batch):

        x, y, categorical_att, continuous_att = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()

        self.optimizer.zero_grad()
        self.d_optimizer.zero_grad()

        train_generator = batch_idx % 3 == 1
        if train_generator:
            outputs = {}

            reconstruction = self.model(categorical_att, continuous_att)
            outputs["x"] = self.oracle(x)
            outputs["reconstruction"] = self.oracle(reconstruction)

            pix_l1 = l1_loss(x, reconstruction)
            feat_l1 = l1_loss(outputs["x"]["z"], outputs["reconstruction"]["z"])

            reconstruction_loss = self.alpha * pix_l1 + (1 - self.alpha) * feat_l1

            d_loss = self.discriminator_loss(reconstruction, real=True)
            g_loss = reconstruction_loss + d_loss * self.lamb

            g_loss.backward()
            self.optimizer.step()

            ret = dict(pix_l1_loss=pix_l1.item(),
                       feat_l1_loss=feat_l1.item(),
                       reconstruction_loss=reconstruction_loss.item(),
                       g_loss=g_loss.item())

        # train discriminator
        else:
            with torch.no_grad():
                reconstruction = self.model(categorical_att, continuous_att)

            fake_loss = self.discriminator_loss(reconstruction, real=False)
            real_loss = self.discriminator_loss(x, real=True)
            d_loss = fake_loss + real_loss
            d_loss.backward()
            self.d_optimizer.step()

            ret = dict(d_loss=d_loss.item())

        return ret


    def val_on_batch(self, epoch, batch_idx, batch, vis_flag):

        x, y, categorical_att, continuous_att = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()
        b = x.size(0)

        outputs = {}
        reconstruction = self.model(categorical_att, continuous_att)
        outputs["x"] = self.oracle(x)
        outputs["reconstruction"] = self.oracle(reconstruction)

        pix_l1 = l1_loss(x, reconstruction)
        feat_l1 = l1_loss(outputs["x"]["z"], outputs["reconstruction"]["z"])

        reconstruction_loss = self.alpha * pix_l1 + (1 - self.alpha) * feat_l1

        real_loss = self.discriminator_loss(x, real=True)
        fake_loss = self.discriminator_loss(reconstruction, real=False)
        d_loss = real_loss + fake_loss

        g_loss = reconstruction_loss + d_loss * self.lamb

        ret = dict(d_loss=d_loss.item(),
                   reconstruction_loss=reconstruction_loss.item(),
                   pix_l1_loss=pix_l1.item(),
                   feat_l1_loss=feat_l1.item(),
                   g_loss=g_loss.item())

        att_mse = l1_loss(outputs["reconstruction"]["pred_continuous"], continuous_att)

        char_acc = (outputs["reconstruction"]["pred_char"].argmax(1) == categorical_att[:, -2]).sum().item() / b
        font_acc = (outputs["reconstruction"]["pred_font"].argmax(1) == categorical_att[:, -1]).sum().item() / b

        ret.update(dict(recontructed_char_acc=char_acc,
                        reconstructed_att_loss=att_mse.item(),
                        reconstructed_font_acc=font_acc))

        if vis_flag and batch_idx == 0 and (epoch + 1) % 10 == 0:
            self.save_img("val_reconstruction", make_grid(x), make_grid(reconstruction), epoch)

        return ret


    def predict_on_batch(self, categorical_att, continuous_att):
        return self.model(categorical_att.cuda(), continuous_att.cuda())


    def train_on_loader(self, epoch, data_loader, pretrained=False):
        """Iterate over the training set
#
        Args:
            data_loader: iterable training data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.n_data = len(data_loader.dataset)
        ret = {}
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        self.model.train()
        self.discriminator.train()


        for i, batch in enumerate(tqdm(data_loader)):
            res_dict = self.train_on_batch(i, batch)
            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        self.scheduler.step()
        return {f"Train/{k}": np.mean(v) for k,v in ret.items()}


    @torch.no_grad()
    def val_on_loader(self, epoch, data_loader, pretrained=False, vis_flag=True):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        self.discriminator.eval()
        ret = {}

        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            res_dict = self.val_on_batch(epoch, batch_idx, batch, vis_flag)
            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        ret = {f"Val/{k}": np.mean(v) for k,v in ret.items()}
        return ret


    def get_state_dict(self):
        ret = {}
        ret["oracle"] = self.oracle.state_dict()
        ret["generator"] = self.model.state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        ret["discriminator"] = self.discriminator.state_dict()
        ret["d_optimizer"] = self.d_optimizer.state_dict()
        ret["scheduler"] = self.scheduler.state_dict()
        ret["char_embedding"] = self.model.char_embedding.state_dict()
        ret["font_embedding"] = self.model.font_embedding.state_dict()

        return ret


    def load_state_dict(self, state_dict):

        self.optimizer.load_state_dict(state_dict["optimizer"])
        # self.oracle.load_state_dict(state_dict["oracle"])
        self.model.load_state_dict(state_dict["generator"])
        self.model.char_embedding.load_state_dict(state_dict["char_embedding"])
        self.model.font_embedding.load_state_dict(state_dict["font_embedding"])
        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.d_optimizer.load_state_dict(state_dict["d_optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
