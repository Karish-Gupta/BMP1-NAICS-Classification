import torch.nn as nn

class HierarchicalLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.3, gamma=0.5):
        super(HierarchicalLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        
        # Weights for Sector, Subsector, and Industry
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, preds, targets):
        # preds: dict from NAICSClassifier (sector, subsector, industry)
        # targets: dict from NAICSDataset labels (sector, subsector, industry)
        
        l1 = self.criterion(preds['sector'], targets['sector'])
        l2 = self.criterion(preds['subsector'], targets['subsector'])
        l3 = self.criterion(preds['industry'], targets['industry'])
        
        return (self.alpha * l1) + (self.beta * l2) + (self.gamma * l3)