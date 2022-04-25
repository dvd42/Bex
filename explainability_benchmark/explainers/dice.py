import torch
import numpy as np
from tqdm import tqdm

from .base import ExplainerBase

torch.backends.cudnn.benchmark = True


class Dice(ExplainerBase):

    def __init__(self, num_explanations=8, proximity_weight=0.5, data_path="data", diversity_weight=1.0, yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist", lr=0.01, max_iters=500, loss_diff_thres=1e-3, init_near_query_instance=True, stopping_threshold=0.5):

        super().__init__()

        self.data_path = data_path
        self.num_explanations = num_explanations
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type
        self.lr = lr
        self.init_near_query_instance = init_near_query_instance
        self.stopping_threshold = stopping_threshold
        self.cache = False
        self.max_iters = max_iters
        self.loss_diff_thres = loss_diff_thres


    def _read_or_write_mads(self):

        if "mads" not in self.loaded_data:
            print("Caching MADs")
            self.write_mads()
        self.read_mads()


    def write_mads(self):

        mads = torch.from_numpy(np.median(abs(self.train_mus - np.median(self.train_mus, 0)), 0))
        to_save = dict(mads=mads)

        for k, v in to_save.items():
            self.loaded_data[k] = v
            print("Done.")


    def read_mads(self):
        print("Reading MADs...")
        self.mads = torch.from_numpy(self.loaded_data['mads'][...])
        print("Done...")

    def explain_batch(self, latents, logits, images, classifier, generator):

        """Generates diverse counterfactual explanations
        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function. Takes "avg_dist" or "dpp_style:inverse_dist".
        :param lr: Learning rate for optimizer.
        :param min_iter: Min iterations to run gradient descent for.
        :param max_iter: Max iterations to run gradient descent for.
        :param loss_diff_thres: Minimum difference between successive loss values to check convergence.
        :param loss_converge_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence. Defaults to 1, but we assigned a more conservative value of 2 in the paper.
        :param verbose: Print intermediate loss value.
        :param init_near_query_instance: Boolean to indicate if counterfactuals are to be initialized near query_instance.
        :param tie_random: Used in rounding off CFs and intermediate projection.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations (see diverse_counterfactuals.py).
        """


        # TODO this is hacky
        if not self.cache:
            self._read_or_write_mads()
            self.cache = True


        mask = torch.ones(latents.shape[1], device=latents.device).unsqueeze(0)

        b, c = latents.shape
        # initialize the cf instances
        if not self.init_near_query_instance:
            cf_instances = torch.rand(b, self.num_explanations, c)
        else: # initialize around the query instances
            cf_instances = latents.repeat(self.num_explanations, 1).view(b, -1, c)
            for i in range(1, self.num_explanations):
                cf_instances[:, i] = cf_instances[:, i] + 0.01 * i

        cf_instances = mask * cf_instances + ((1 - mask) * latents)[:, None, :]

        # self.total_CFs = total_CFs
        # self.yloss_type = yloss_type
        # self.diversity_loss_type = diversity_loss_type

        # self.feature_weights_list = np.ones((1, query_instance.shape[1]))
        mads = self.mads.cuda()

        inverse_mads = 1 / mads
        # inverse_mads = np.round(inverse_mads, 2)
        self.feature_weights_list = inverse_mads

        optimizer = torch.optim.Adam([cf_instances], lr=self.lr)


        self.target_cf_class = 1 - logits.max(1)[1]
        self.target_cf_class = self.target_cf_class.repeat(self.num_explanations)

        # self.min_iter = min_iter
        # self.max_iter = max_iter
        # self.loss_diff_thres = loss_diff_thres
        # no. of iterations to wait to confirm that loss has converged

        if self.stopping_threshold != 0.5:
            self.stopping_threshold = torch.zeros_like(self.target_cf_class).fill_(0.75)
            self.stopping_threshold[self.target_cf_class == 0] = 0.25
        # if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
        #     self.stopping_threshold = 0.25
        # elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
        #     self.stopping_threshold = 0.75

        # running optimization steps
        # start_time = time.time()

        # looping the find CFs depending on whether its random initialization or not
        iterations = 0
        loss_diff = float("inf")
        prev_loss = 0
        logits = None
        pbar = tqdm(total=self.max_iters)
        while self.stop_loop(iterations, loss_diff, logits) is False:

            cf_instances.requires_grad = True
            optimizer.zero_grad()

            proximity_loss = 0
            diversity_loss = 0
            decoded = generator(cf_instances.view(-1, c))
            logits = classifier(decoded)

            yloss = self.compute_yloss(logits.max(1)[0])
            if self.proximity_weight > 0:
                proximity_loss = self.compute_proximity_loss(latents, cf_instances)
            if self.diversity_weight > 0:
                diversity_loss = self.compute_diversity_loss(cf_instances)

            loss_value = self.compute_loss(yloss, proximity_loss, diversity_loss)
            loss_value.backward()

            cf_instances.grad = cf_instances.grad * mask
            # update the variables
            optimizer.step()

            loss_diff = abs(loss_value-prev_loss)
            prev_loss = loss_value
            iterations += 1
            pbar.update(1)



        return cf_instances


    def compute_yloss(self, logits):

        if self.yloss_type == 'l2_loss':
            yloss = torch.pow((logits - self.target_cf_class), 2)
        elif self.yloss_type == "log_loss":
            logits = torch.log(torch.abs(logits - 1e-6) / torch.abs(1 - logits - 1e-6))
            criterion = torch.nn.BCEWithLogitsLoss()
            yloss = criterion(logits, self.target_cf_class)

        elif self.yloss_type == 'hinge_loss':
            logits = torch.log(torch.abs(logits - 1e-6) / torch.abs(1 - logits - 1e-6))
            criterion = torch.nn.ReLU()
            all_ones = torch.ones_like(self.target_cf_class)
            labels = 2 * self.target_cf_class - all_ones
            temp_loss = all_ones - torch.mul(labels, logits)
            yloss = criterion(temp_loss)

        else:
            raise ValueError(f"yloss type {self.yloss_type} not supported")

        return yloss.mean()


    def compute_proximity_loss(self, latents, cf_instances):
        """compute weighted distance between query intances and counterfactual explanations"""
        return torch.mean(torch.abs(cf_instances - latents[:, None, :]) * self.feature_weights_list)


    def compute_dist(self, x1, x2):
        return torch.sum(torch.mul(torch.abs(x1 - x2), self.feature_weights_list), dim=0).mean()


    def dpp_style(self, cf_instances, submethod):
        """Computes the DPP of a matrix."""

        det_entries = torch.ones(self.num_explanations, self.num_explanations)
        for i in range(self.num_explanations):
            for j in range(self.num_explanations):
                det_entries[i, j] = self.compute_dist(cf_instances[:, i], cf_instances[:, j])

        if submethod == "inverse_dist":
            det_entries = 1.0 / (1.0 + det_entries)
        elif submethod == "exponential_dist":
            det_entries = 1.0 / (torch.exp(det_entries))
        else:
            raise ValueError(f"diversity_loss_type sub-method {submethod} not supported")

        det_entries += torch.eye(self.num_explanations) * 0.0001
        return torch.det(det_entries)


    def compute_diversity_loss(self, cf_instances):
        """Computes the third part (diversity) of the loss function."""
        if self.num_explanations == 1:
            return torch.tensor(0.0)

        if "dpp" in self.diversity_loss_type:
            submethod = self.diversity_loss_type.split(':')[1]
            return self.dpp_style(cf_instances, submethod)
        elif self.diversity_loss_type == "avg_dist":
            return 1 - 1 / (1.0 + self.mm(cf_instances, cf_instances.T))

        else:
            raise ValueError(f"diversity_loss_type {self.diversity_loss_type} not supported")

    # def compute_regularization_loss(self, cf_instances):
    #     """Adds a linear equality constraints to the loss functions - to ensure all levels of a categorical variable sums to one"""
    #     regularization_loss = 0
    #     # for v in self.data_interface.encoded_categorical_feature_indices:
    #     for v in [[-2, -1]]:
    #         regularization_loss += torch.sum(torch.pow((torch.sum(cf_instances[:, v[0]:v[-1]+1], axis = 1) - 1.0), 2))

    #     return regularization_loss


    def compute_loss(self, yloss, proximity_loss, diversity_loss):
        """Computes the overall loss"""
        # yloss = self.compute_yloss(cf_instances)
        # proximity_loss = self.compute_proximity_loss(query_instance, cf_instances) if proximity_weight > 0 else 0.0
        # diversity_loss = self.compute_diversity_loss(cf_instances) if diversity_weight > 0 else 0.0
        # regularization_loss = self.compute_regularization_loss(cf_instances)

        # loss = yloss + (proximity_weight * proximity_loss) - (diversity_weight * diversity_loss) + (categorical_penalty * regularization_loss)
        loss = yloss + (self.proximity_weight * proximity_loss) - (self.diversity_weight * diversity_loss)
        return loss


    def stop_loop(self, itr, loss_diff, test_preds):
        """Determines the stopping condition for gradient descent."""

        # stop GD if max iter is reached
        if itr >= self.max_iters:
            return True

        # else stop when loss diff is small & all CFs are valid (less or greater than a stopping threshold)
        if loss_diff <= self.loss_diff_thres:
            test_preds = torch.sigmoid(test_preds)
            test_preds = torch.where(self.target_cf_class == 0, test_preds[:, 0], test_preds[:, 1])
            if (test_preds > self.stopping_threshold).all():
                return True
                # if self.target_cf_class == 0 and (test_preds < self.stopping_threshold).all():
                #     return True
            # if self.target_cf_class == 1 and (test_preds > self.stopping_threshold).all():
                # return True
            return False

        return False
