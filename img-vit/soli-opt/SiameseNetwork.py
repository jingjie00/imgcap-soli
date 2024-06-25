import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
        
    def forward_one(self, x):
        outputs = self.encoder(x).last_hidden_state
        return outputs
    
    def forward(self, x1, x2):
        print(f"forward input shapes: {x1.shape}, {x2.shape}")  # Debugging line
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        return output1, output2


# Define the contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
