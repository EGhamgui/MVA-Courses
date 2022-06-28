import torch
from torch import nn
import torchvision.models as models
from fastai.vision import AdaptiveConcatPool2d, Flatten

# -------------------------------------------------------------------------------------------------- #

class pooling_ResNet18(nn.Module):
    '''
    Reference: https://www.kaggle.com/code/iafoss/panda-concat-tile-pooling-starter-0-79-lb/notebook
    
    '''

    def __init__(self, k = 1028, n=6):
        super().__init__()
        m = models.resnet18(pretrained=True)
        self.enc = nn.Sequential(*list(m.children())[:-2])   
        nc = list(m.children())[-1].in_features
        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                  Flatten(),
                                  nn.Linear(2*nc,k),
                                  nn.Mish(),
                                  nn.BatchNorm1d(k), 
                                  nn.Dropout(0.5),
                                  nn.Linear(k,n)
                                  )
        
    def forward(self, *x):
        shape = x[0].shape
        n = len(x)

        # 1.       
        x = torch.stack(x,1).view(-1,shape[1],shape[2],shape[3])
        #x: bs*N x 3 x 128 x 128

        # 2.
        x = self.enc(x)
        #x: bs*N x C x 4 x 4

        # 3.
        x = x.view(-1,n,x.shape[1],x.shape[2],x.shape[3]).permute(0,2,1,3,4).contiguous().view(-1,x.shape[1],x.shape[2]*n,x.shape[3])
        #x: bs x C x N*4 x 4        

        # 4.
        x = self.head(x)
        #x: bs x n

        return x