import torch
import numpy as np


class DistAlignEMA():
    def __init__(self, num_classes=2, momentum=0.999, p_target=None):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum

        # p_target
        self.update_p_target, self.p_target = self.set_p_target(p_target)    
        print('distribution alignment p_target:', self.p_target)
        # p_model
        self.p_model = None

    def dist_align(self, probs_x_ulb):
        # update queue
        # probs_x_ulb:(64,10)  probs_x_lb:(64,10) one-hot编码，soft label
        self.update_p(probs_x_ulb)

        # dist align
        # self.p_target [0.1, ..., 0.1]
        # self.p_model:(1,10) probs_x_ulb在0维上的平均
        # probs_x_ulb_aligned:(64,10)
        # import pdb; pdb.set_trace()
        probs_x_ulb_aligned = probs_x_ulb * (self.p_target + 1e-6) / (self.p_model + 1e-6)
        probs_x_ulb_aligned = probs_x_ulb_aligned / probs_x_ulb_aligned.sum(dim=-1, keepdim=True)
        return probs_x_ulb_aligned

    def update_p(self, probs_x_ulb):
        # check device
        if not self.p_target.is_cuda:
            self.p_target = self.p_target.to(probs_x_ulb.device)

        probs_x_ulb = probs_x_ulb.detach()
        self.p_model = torch.mean(probs_x_ulb, dim=0)
    
    def set_p_target(self,  p_target=None):
        # p_target
        """
        tensor([0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000])
        """
        update_p_target = False
        p_target = torch.ones((self.num_classes, )) / self.num_classes

        return update_p_target, p_target
    