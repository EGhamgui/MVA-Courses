import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

# -------------------------------------------------------------------------------------------------- #

class MIL_resnet18(nn.Module):

  def __init__(self):
      super().__init__()
      self.L = 3000
      self.D = 64
      self.K = 1

      self.feature_extractor_part1 = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=3),
          nn.ReLU(),
          nn.MaxPool2d(2, stride=2))

      self.model_pre = models.resnet18(pretrained=True)
      self.pre_layers1 = nn.Sequential(*list(self.model_pre.children())[2:6])

      self.attention = nn.Sequential(
          nn.Linear(32768, self.D),
          nn.Tanh(),
          nn.Linear(self.D, self.K))

      self.classifier = nn.Sequential(
          nn.Linear(32768*self.K, 6))

  def forward(self, x):
      x = x.squeeze(0)
      H = self.feature_extractor_part1(x)
      H = self.pre_layers1(H)
      H = H.view(-1, 32768)
      A = self.attention(H)  
      A = torch.transpose(A, 1, 0)  
      A = F.softmax(A, dim=1)   
      M = torch.mm(A, H)  
      
      Y_prob = self.classifier(M)
      Y_hat = Y_prob.float()

      return Y_hat