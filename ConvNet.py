import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode=1, debug=False):
        super(ConvNet, self).__init__()
        self.debug = debug

        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, assign one of the valid mode.
        # This will fix the forward function (and the network graph) for training/testing
        if mode == 1:
            # Uses square kernels of size 3x3 with stride 1
            # Include padding of size P = (K-1)/2 to preserve dimensions 
            self.conv1 = nn.Conv2d(in_channels=3,
                                   out_channels=8,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

            self.conv2 = nn.Conv2d(in_channels=8,
                                   out_channels=16,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            self.conv3 = nn.Conv2d(in_channels=16,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

            self.conv4 = nn.Conv2d(in_channels=32,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            self.conv5 = nn.Conv2d(in_channels=64,
                                   out_channels=128,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            # Add dropout to randomly make some pixels 0 with the probability p
            # Use dropout2d since it is a spatial-dropout designed for 4-D tensors such as images or feature maps from convolution layers. The standard dropout will not be able to effectively regularize the network.
            self.dropout = nn.Dropout2d(p=0.5)
            
            # COCO images have possible 91 category ids to predict
            # Define fcNN which performs the classification
            self.fc1 = nn.Linear(128 * 20 * 20, 1024)
            self.fc2 = nn.Linear(1024, 92)
            self.forward = self.model_1

        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
    def model_1(self, X):
        if self.debug: print("model1 input:\t", X.shape) 
            
        X = F.relu(self.conv1(X))
        if self.debug: print("conv1:\t\t", X.shape)
        
        X = F.max_pool2d(X, 2)
        if self.debug: print("max_pool1:\t", X.shape)
        
        X = F.relu(self.conv2(X))
        if self.debug: print("conv2:\t\t", X.shape)
        
        X = F.max_pool2d(X, 2)
        if self.debug: print("max_pool2:\t", X.shape)
        
        X = F.relu(self.conv3(X))
        if self.debug: print("conv3:\t\t", X.shape)
        
        X = F.max_pool2d(X, 2)
        if self.debug: print("max_pool3:\t", X.shape)

        X = F.relu(self.conv4(X))
        if self.debug: print("conv4:\t\t", X.shape)
        
        X = F.max_pool2d(X, 2)
        if self.debug: print("max_pool4:\t", X.shape)

        X = F.relu(self.conv5(X))
        if self.debug: print("conv5:\t\t", X.shape)
        
        X = F.max_pool2d(X, 2)
        if self.debug: print("max_pool5:\t", X.shape)
            
        X = self.dropout(X)
        if self.debug: print("dropout:\t", X.shape)
        
        X = torch.flatten(X, start_dim=1)
        if self.debug: print("flatten:\t", X.shape)
            
        X = F.relu(self.fc1(X))
        if self.debug: print("fc1:\t\t", X.shape)
        
        output = torch.sigmoid(self.fc2(X))
        if self.debug: print("fc2:\t\t", X.shape)
            
        return output