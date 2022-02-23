from abc import abstractmethod
import os
from tqdm import tqdm
import numpy as np
import h5py
import torch

class ExplainerBase:

    def __init__(self, data_path="data"):

        super().__init__()
        self.data_path = data_path
        self.num_classes = 2


        digest = "cache_fim_new.hdf"
        self.digest = os.path.join(self.data_path, digest)


    def read_cache(self):

        print("Loading from %s" % self.digest)
        self.loaded_data = h5py.File(self.digest, 'a')
        try:
            self.logits = torch.from_numpy(self.loaded_data['logits'][...])
            self.mus = torch.from_numpy(self.loaded_data['mus'][...])
            self.train_mus = torch.from_numpy(self.loaded_data['train_mus'][...])
            self.latent_mean = self.train_mus.mean()
            self.latent_std = self.train_mus.std()
            # normalize latents with the training statistics
            self.mus = (self.mus - self.latent_mean) / self.latent_std

        except:
            print("Data not found in the hdf5 cache")


    def write_cache(self, loader, encoder, generator, classifier, prefix):

        """Loops through the data and stores latents """

        preds = []
        mus = []
        for idx, x, y, categorical_att, continuous_att in tqdm(loader):
            with torch.no_grad():
                x = x.cuda()
                categorical_att = categorical_att.cuda()
                continuous_att = continuous_att.cuda()
                z = encoder(categorical_att, continuous_att)
                mus.append(z.cpu().numpy())

                if "train" not in prefix:
                    reconstruction = generator(z)
                    logits = classifier(reconstruction)
                    preds.append(logits.data.cpu().numpy())

        to_save = dict(mus=np.concatenate(mus, 0))
        if "train" not in prefix:
            to_save["logits"] = np.concatenate(preds, 0)

        with h5py.File(self.digest, 'a') as outfile:
            for k, v in to_save.items():
                outfile[f"{prefix}_{k}"] = v
                print("Done.")

    def get_latents(self, idx):

        return self.mus[idx].cuda()

    def get_logits(self, idx):

        return self.logits[idx].cuda()


    def cleanup(self):

        self.loaded_data.close()


    @abstractmethod
    def explain_batch(self, latents, logits, images, labels, classifier, generator):
        raise NotImplementedError