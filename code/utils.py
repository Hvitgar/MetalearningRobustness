import torch
import torch.nn.functional as F

class GumbelSoftmax(torch.distributions.Distribution):
    def __init__(self, temperature, logits):
        self.temperature = temperature
        self.logits = logits
        self.G = torch.distributions.Gumbel(0, 1)
    
    def rsample(self, sample_shape, hard=False):
        if type(sample_shape) == list:
            sample_shape = torch.Size(sample_shape)
        g = self.G.sample(sample_shape + self.logits.size()).to(self.logits.device)
        y = F.softmax((F.log_softmax(self.logits) + g) / self.temperature, dim=-1)
        if hard:
            shape = y.size()
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            y = (y_hard - y).detach() + y
        return y