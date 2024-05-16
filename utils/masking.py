import torch


class SoftMatchWeighting():
    """
    SoftMatch learnable truncated Gaussian weighting
    """
    def __init__(self, num_classes=2, n_sigma=2, momentum=0.999):
        self.num_classes = num_classes
        self.n_sigma = n_sigma
        self.m = momentum

        # initialize Gaussian mean and variance
        self.prob_max_mu_t = torch.tensor(1.0 / self.num_classes)
        self.prob_max_var_t = torch.tensor(1.0)

    def update(self, probs_x_ulb):
        # 长度为64的张量，batch中64个无标签样本
        # 这里的max_probs也就是文章中的max(p)
        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        # import pdb; pdb.set_trace()
        prob_max_mu_t = torch.mean(max_probs)  # torch.quantile(max_probs, 0.5)
        prob_max_var_t = torch.var(max_probs, unbiased=True)
        self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t.item()   # μt
        self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t.item()  # δt

        return max_probs, max_idx
    
    def masking(self, logits_x_ulb, softmax_x_ulb=True):
        # 把数据转到cuda上
        if not self.prob_max_mu_t.is_cuda:
            self.prob_max_mu_t = self.prob_max_mu_t.to(logits_x_ulb.device)
        if not self.prob_max_var_t.is_cuda:
            self.prob_max_var_t = self.prob_max_var_t.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        # compute weight
        mu = self.prob_max_mu_t
        var = self.prob_max_var_t

        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / (self.n_sigma ** 2))))
        # print('+++++++++')
        # print(self.prob_max_mu_t.item())
        # print(torch.min(max_probs).item())
        # print(torch.min(mask).item())
        # print('+++++++++')
        return mask