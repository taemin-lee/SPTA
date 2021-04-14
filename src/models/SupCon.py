
import torch.nn as nn
import torch.nn.functional as F

model_dict = {
    'resnet10': 512,
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'mobilenet': 1024,
    'wideres': 640,
}

class SupConNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, model, name='resnet18', feat_dim=128):
        super(SupConNet, self).__init__()
        dim_in = model_dict[name]
        self.encoder = model
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim)
        )

    def forward(self, x):
        feat = self.encoder(x, feature=True)[0]
        feat = F.normalize(self.head(feat), dim=1)
        return feat


