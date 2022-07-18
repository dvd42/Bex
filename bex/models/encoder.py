import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .backbones import get_backbone


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


class Encoder(torch.nn.Module):

    def __init__(self, exp_dict, writer=None):
        """ Constructor
        Args:
            model: architecture to train
            self.exp_dict: reference to dictionary with the global state of the application
        """
        super().__init__()
        self.model = get_backbone(exp_dict)
        self.exp_dict = exp_dict
        self.ngpu = self.exp_dict["ngpu"]
        self.z_dim = exp_dict["z_dim"]
        self.w = self.exp_dict["dataset"]["width"]
        self.h = self.exp_dict["dataset"]["height"]

        self.model.cuda()

        if self.ngpu > 1:
            self.model = torch.nn.DataParallel(self.model, list(range(self.ngpu)))

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                          betas=(0.9, 0.999),
                                          lr=self.exp_dict["lr"], weight_decay=exp_dict["weight_decay"])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=exp_dict["max_epoch"])


    def train_on_batch(self, batch):

        x, y, categorical_att, continuous_att = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()
        b = x.size(0)

        outputs = self.model(x)

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


    def val_on_batch(self, batch):

        x, y, categorical_att, continuous_att = batch
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        categorical_att = categorical_att.cuda()
        continuous_att = continuous_att.cuda()
        b = x.size(0)

        outputs = self.model(x)

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

        for batch in tqdm(data_loader):
            res_dict = self.train_on_batch(batch)
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
        ret = {}

        for batch in tqdm(data_loader):
            res_dict = self.val_on_batch(batch)
            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        ret = {f"Val/{k}": np.mean(v) for k,v in ret.items()}
        return ret


    def get_state_dict(self):
        ret = {}
        ret["model"] = self.model.state_dict()
        ret["optimizer"] = self.optimizer.state_dict()

        return ret


    def load_state_dict(self, state_dict):

        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.model.load_state_dict(state_dict["model"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
