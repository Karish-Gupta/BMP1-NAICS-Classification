import torch.nn as nn

class HierarchicalLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super(HierarchicalLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        
        # Weights for Sector, Subsector, and Industry
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, preds, targets):
        sector_logits, subsector_logits, industry_logits = preds
        sector_labels, subsector_labels, industry_labels = targets
        
        l1 = self.criterion(sector_logits, sector_labels)
        l2 = self.criterion(subsector_logits, subsector_labels)
        l3 = self.criterion(industry_logits, industry_labels)
        
        return (self.alpha * l1) + (self.beta * l2) + (self.gamma * l3)