import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsensusLoss(nn.Module):
    def __init__(self, nClass, div):
        super(ConsensusLoss, self).__init__()
        self.nClass = nClass
        self.div = div
        
    def forward(self, x, y):
        if self.div == 'kl':
            x = F.softmax(x, dim=1)
            y = F.log_softmax(y, dim=1)
            kl_div = F.kl_div(y, x, reduction='batchmean') #x 
            
            return kl_div
        elif self.div == 'kl_d':
            x = F.softmax(x, dim=1)
            y = F.log_softmax(y, dim=1)
            x_d = x.detach()
            kl_div = F.kl_div(y, x_d, reduction='batchmean') #detached x
            
            return kl_div
        elif self.div == 'l1':
            x = F.softmax(x, dim=1)
            y = F.softmax(y, dim=1) 
            l1_div = (x - y).abs().mean(1).mean() #l1 norm
            
            return l1_div
        elif self.div == 'l2':
            x = F.softmax(x, dim=1)
            y = F.softmax(y, dim=1)
            l2_div = (x - y).pow(2).sum(1).sqrt().mean() #l2 norm
            
            return l2_div
        elif self.div == 'neg_cos':
            x = F.softmax(x, dim=1)
            y = F.softmax(y, dim=1)
            
            neg_cos_div = 0.5 * (1 - ((x * y).sum(1) / x.norm(2, dim=1) / y.norm(2, dim=1))).mean()
            
            return neg_cos_div
        
        
        