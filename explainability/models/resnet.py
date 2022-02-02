import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn.functional as F
from backbones import get_backbone

class ResNet(torch.nn.Module):
    def __init__(self, exp_dict):
        super().__init__()

        self.exp_dict = exp_dict
        self.model = get_backbone(exp_dict, labelset=None)
        self.model.cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), betas=(0.9, 0.999),
                                           lr=self.exp_dict["lr"], weight_decay=self.exp_dict["weight_decay"])


    def train_on_loader(self, epoch, data_loader, pretrained=False):

        self.model.train()
        ret = {}

        for i, batch in enumerate(tqdm(data_loader)):
            res_dict = self.train_on_batch(i, batch)
            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        return {f"Train/{k}": np.mean(v) for k,v in ret.items()}


    @torch.no_grad()
    def val_on_loader(self, epoch, data_loader, vis_flag=True, pretrained=False):

        self.model.eval()
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


    def train_on_batch(self, epoch, batch):

        x, y = batch
        b = x.size(0)

        x = x.cuda()
        y = y.cuda()

        self.optimizer.zero_grad()

        out = self.model(x)

        loss = F.cross_entropy(out, y)
        loss.backward()
        self.optimizer.step()

        acc = (out.argmax(1) == y).sum() / b

        return dict(loss=loss.item(), accuracy=float(acc))


    def val_on_batch(self, epoch, batch_idx, batch, vis_flag):

        x, y = batch
        b = x.size(0)

        x = x.cuda()
        y = y.cuda()

        out = self.model(x)

        loss = F.cross_entropy(out, y)

        acc = (out.argmax(1) == y).sum() / b

        return dict(loss=loss.item(), accuracy=float(acc))


    def get_state_dict(self):

        ret = {}
        ret["model"] = self.model.state_dict()
        ret["optimizer"] = self.optimizer.state_dict()

        return ret


    def load_state_dict(self, state_dict):

        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.model.load_state_dict(state_dict["model"])
