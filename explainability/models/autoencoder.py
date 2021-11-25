import random
import wandb
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.losses import get_kl_loss, l1_loss
from utils.metrics import _compute_sap
from tqdm import tqdm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.parallel as parallel
from backbones.biggan import Discriminator, DiscriminatorLoss
from backbones import get_backbone


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


class Autoencoder(torch.nn.Module):
    """Trains an Autoencoder on multiple GPUs"""

    def __init__(self, exp_dict, labelset, writer=None):
        """ Constructor
        Args:
            model: architecture to train
            self.exp_dict: reference to dictionary with the global state of the application
        """
        super().__init__()
        self.model = get_backbone(exp_dict, labelset)
        self.exp_dict = exp_dict
        self.ngpu = self.exp_dict["ngpu"]
        self.devices = list(range(self.ngpu))
        self.labelset = labelset
        # self.is_categorical = {}
        self.z_dim = exp_dict["z_dim"]
        self.w = self.exp_dict["dataset"]["width"]
        self.h = self.exp_dict["dataset"]["height"]
        self.alpha = exp_dict["alpha"]
        self.lamb = exp_dict["lambda"]

        if min(self.w, self.h) == 128:
            self.ratio = 32
        elif min(self.w, self.h) == 64:
            self.ratio = 16
        elif min(self.w, self.h) == 32:
            self.ratio = 8

        self.output_w = self.w // self.ratio
        self.output_h = self.h // self.ratio
        self.channels_width = exp_dict["backbone"]["channels_width"]


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
                                          lr=self.exp_dict["lr"], weight_decay=exp_dict["weight_decay"])

        self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(),
                                          betas=(0.9, 0.999),
                                          lr=self.exp_dict["lr"], weight_decay=exp_dict["weight_decay"])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=exp_dict["max_epoch"])


    def save_img(self, name, images, reconstructions, idx=0):

        f, ax = plt.subplots(1, 2)
        f.set_size_inches(15.5, 15.5)
        ax[0].imshow(images.cpu().permute(1, 2, 0))
        ax[1].imshow(reconstructions.cpu().permute(1, 2, 0))
        ax[0].axis('off')
        ax[1].axis('off')

        wandb.log({name: f}, commit=False)
        plt.close()


    def train_encoder_on_batch(self, batch_idx, batch):

        x, y, categorical_att, continuous_att = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()
        b = x.size(0)

        outputs = self.model.encode(x)

        att_mse = l1_loss(outputs["pred_continuous"], continuous_att)
        font_loss = F.cross_entropy(outputs["pred_font"], categorical_att[:, -1])
        char_loss = F.cross_entropy(outputs["pred_char"], categorical_att[:, -2])
        ce_loss = (font_loss + char_loss) / 2
        loss = att_mse + ce_loss
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        font_acc = (outputs["pred_font"].argmax(1) == categorical_att[:, -1]).sum().item() / b
        char_acc = (outputs["pred_char"].argmax(1) == categorical_att[:, -2]).sum().item() / b

        ret = dict(char_acc=char_acc,
                   font_acc=font_acc,
                   att_l1_loss=att_mse.item(),
                   ce_loss=ce_loss.item(),
                   loss=loss.item())

        return ret


    def train_on_batch(self, batch_idx, batch):

        x, y, categorical_att, continuous_att = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()

        # import cv2
        # mean = torch.tensor([0.5] * 3)[None, :, None, None]
        # new_x = x.cpu() * mean + mean

        # blacks = new_x[continuous_att[:, 0] == 1]
        # whites = new_x[continuous_att[:, 0] == 0]

        # from torchvision.utils import make_grid

        # white_grid = make_grid(whites).permute(1, 2, 0).numpy()
        # black_grid = make_grid(blacks).permute(1, 2, 0).numpy()

        # cv2.imshow("White Letters", white_grid)
        # cv2.imshow("Black Letters", black_grid)

        # # for i_x, i_y in zip(new_x, continuous_att):

        #     # print(y[0].item())
        #     # cv2.imshow("img", x.permute(1, 2, 0).numpy())
        # cv2.waitKey(0)

        self.optimizer.zero_grad()
        self.d_optimizer.zero_grad()

        train_generator = batch_idx % 3 == 1
        if train_generator:

            outputs, reconstruction = self.model(x, categorical_att, continuous_att)

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
                outputs, reconstruction = self.model(x, categorical_att, continuous_att)

            fake_loss = self.discriminator_loss(reconstruction, real=False)
            real_loss = self.discriminator_loss(x, real=True)
            d_loss = fake_loss + real_loss
            d_loss.backward()
            self.d_optimizer.step()

            ret = dict(d_loss=d_loss.item())

        return ret


    def val_encoder_on_batch(self, epoch, batch_idx, batch, vis_flag):

        x, y, categorical_att, continuous_att = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()
        b = x.size(0)

        outputs = self.model.encode(x)

        att_mse = l1_loss(outputs["pred_continuous"], continuous_att)
        font_loss = F.cross_entropy(outputs["pred_font"], categorical_att[:, -1])
        char_loss = F.cross_entropy(outputs["pred_char"], categorical_att[:, -2])

        ce_loss = (font_loss + char_loss) / 2
        loss = ce_loss
        loss += att_mse

        font_acc = (outputs["pred_font"].argmax(1) == categorical_att[:, -1]).sum().item() / b
        char_acc = (outputs["pred_char"].argmax(1) == categorical_att[:, -2]).sum().item() / b

        ret = dict(att_l1_loss=att_mse.item(),
                   font_acc=font_acc,
                   char_acc=char_acc,
                   ce_loss=ce_loss.item(),
                   loss=loss.item())

        return ret


    def val_on_batch(self, epoch, batch_idx, batch, vis_flag):

        x, y, categorical_att, continuous_att = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()
        b = x.size(0)

        outputs, reconstruction = self.model(x, categorical_att, continuous_att)

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


    def predict_on_batch(self, x):
        return self.model(x.cuda())


    def train_on_loader(self, epoch, data_loader):
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

        if self.exp_dict["encoder_weights"] is None:
            train_func = self.train_encoder_on_batch
        else:
            train_func = self.train_on_batch

            # Freeze encoder
            for name, p in self.model.named_children():
                if "encoder" in name:
                    getattr(self.model, name).eval()
                    p.requires_grad_(False)

        for i, batch in enumerate(tqdm(data_loader)):
            res_dict = train_func(i, batch)
            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        self.scheduler.step()
        return {f"Train/{k}": np.mean(v) for k,v in ret.items()}


    @torch.no_grad()
    def val_on_loader(self, epoch, data_loader, vis_flag=True):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        self.discriminator.eval()
        ret = {}

        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        val_func = self.val_on_batch if self.pretrained else self.val_encoder_on_batch
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            res_dict = val_func(epoch, batch_idx, batch, vis_flag)
            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        ret = {f"Val/{k}": np.mean(v) for k,v in ret.items()}
        return ret

    @torch.no_grad()
    def test_on_loader(self, data_loader, max_iter=None):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        test_loss_meter = BasicMeter.get("test_loss").reset()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(data_loader):
            mse, regularizer, loss = self.val_on_batch(batch_idx, batch, False)
            test_loss_meter.update(float(loss), 1)
        return {"test_loss": test_loss_meter.mean()}

    def get_state_dict(self):
        ret = {}
        ret["encoder"] = self.model.encoder.state_dict()
        ret["decoder"] = self.model.decoder.state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        ret["discriminator"] = self.discriminator.state_dict()
        ret["d_optimizer"] = self.d_optimizer.state_dict()
        ret["scheduler"] = self.scheduler.state_dict()

        return ret

    def reinitialize_optim(self):

        self.optimizer = torch.optim.AdamW(self.model.parameters(), betas=(0.9, 0.999),
                                           lr=self.exp_dict["lr"], weight_decay=self.exp_dict["weight_decay"])
        self.d_optimizer = torch.optim.AdamW(self.discriminator.parameters(), betas=(0.9, 0.999),
                                             lr=self.exp_dict["lr"], weight_decay=self.exp_dict["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.exp_dict["max_epoch"])


    def load_encoder(self, state_dict):
        self.model.encoder.load_state_dict(state_dict["encoder"])


    def load_state_dict(self, state_dict):

        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.model.encoder.load_state_dict(state_dict["encoder"])
        self.model.decoder.load_state_dict(state_dict["decoder"])
        self.discriminator.load_state_dict(state_dict["discriminator"])
        self.d_optimizer.load_state_dict(state_dict["d_optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])

    def get_lr(self):
        ret = {}
        for i, param_group in enumerate(self.optimizer.param_groups):
            ret["current_lr_%d" % i] = float(param_group["lr"])
        return ret

    def is_end_of_training(self):
        lr = self.get_lr()["current_lr_0"]
        return lr <= (self.exp_dict["lr"] * self.exp_dict["min_lr_decay"])
