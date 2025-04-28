import torch
from torch import nn


class Restocker(nn.Module):
    def __init__(self):
        super().__init__()
        #layering goes here for the convolution stack
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #Looking for features with 16 filters
            nn.BatchNorm2d(32),
            nn.ReLU(), #Relu

            nn.Conv2d(32, 64, kernel_size=3, padding=1), #32x64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
            
        )
         #finds the size of the flattened output from the convolution step
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 128)
            dummy_output = self.conv_stack(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).shape[1]

        self.flatten = nn.Flatten() #Flattens the output into a single layer input

        #Linear classification stack
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 52),
        )
    
    #forward propogation
    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
########################################################################
# END OF MODEL DEFENITION #
########################################################################