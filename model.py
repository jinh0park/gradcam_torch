import torch
import torch.nn as nn
import torch.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ZeroNet(nn.Module):
    '''
    Model architecture is almost same to VGGnet-11.
    '''
    def __init__(self, num_classes):
        # Way more parameters than original model
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2),

            nn.Conv2d(64,128,3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2),

            nn.Conv2d(128,256,3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(12*12*256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_n_params(self):
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp
