import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode=1, debug=False, num_cats=92):
        super(ConvNet, self).__init__()
        self.debug = debug

        # This will select the forward pass function based on mode for the ConvNet.
        # 3 modes are available for evaluation.
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
            # Use dropout2d since it is a spatial-dropout designed for 4-D tensors
            # such as images or feature maps from convolution layers. The standard 
            # dropout will not be able to effectively regularize the network.
            self.dropout = nn.Dropout2d(p=0.5)
            
            # Define fcNN which performs the classification
            # COCO images have possible 91 category ids to predict
            # Since we are not using all 92 categories, 
            # set the size of the last layer to be the number of categories.
            self.fc1 = nn.Linear(128 * 4 * 4, 1028)
            self.fc2 = nn.Linear(1028, num_cats)
            self.forward = self.model_1
            
        elif mode == 2:
            # Encoder structure inspired by SegNET 
            # Conv1 
            self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
            self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
            
            # Conv2 
            self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
            self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
            
            # Conv3 
            self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
            self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
            self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
            
            # Conv4 
            self.conv4_1 = nn.Conv2d(256, 512, 3, 1 , 1)
            self.conv4_2 = nn.Conv2d(512, 512, 3, 1 , 1)
            self.conv4_3 = nn.Conv2d(512, 512, 3, 1 , 1)
            
            # Conv fc5
            self.fc5 = nn.Conv2d(512, 4096, 8)
            self.dropout5 = nn.Dropout2d(p=0.5)
            
            # Conv fc6
            self.fc6 = nn.Conv2d(4096, 4096, 1)
            self.dropout6 = nn.Dropout2d(p=0.5)
            
            # Conv fc7
            self.fc7 = nn.Conv2d(4096, num_cats, 1)
            self.forward = self.model_2
            
        elif mode == 3:
            # Encoder Conv layers
            self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
            self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
            self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
            self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
            self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
            
            self.dropout = nn.Dropout2d(p=0.5)
            
            # Fully connected conv2d layers
            self.fc6 = nn.Conv2d(128, 128, 4)
            # Prediction layer
            self.fc7 = nn.Conv2d(128, num_cats, 1)
            self.forward = self.model_3
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

    def model_2(self, X):
        Y = X
        if self.debug: print("model1 input:\t", X.shape) 
            
        # Conv1
        Y = F.relu(self.conv1_1(Y))
        if self.debug: print("conv1_1:\t\t", Y.shape)
        
        Y = F.relu(self.conv1_2(Y))
        if self.debug: print("conv1_2:\t\t", Y.shape)
            
        Y = F.max_pool2d(Y, 2)
        if self.debug: print("max_pool1:\t", Y.shape)
        
        # Conv2
        Y = F.relu(self.conv2_1(Y))
        if self.debug: print("conv2_1:\t\t", Y.shape)
        
        Y = F.relu(self.conv2_2(Y))
        if self.debug: print("conv2_2:\t\t", Y.shape)
            
        Y = F.max_pool2d(Y, 2)
        if self.debug: print("max_pool2:\t", Y.shape)
            
        # Conv3
        Y = F.relu(self.conv3_1(Y))
        if self.debug: print("conv3_1:\t\t", Y.shape)
        
        Y = F.relu(self.conv3_2(Y))
        if self.debug: print("conv3_2:\t\t", Y.shape)

        Y = F.relu(self.conv3_3(Y))
        if self.debug: print("conv3_2:\t\t", Y.shape)
            
        Y = F.max_pool2d(Y, 2)
        if self.debug: print("max_pool3:\t", Y.shape)
        
        # Conv4
        Y = F.relu(self.conv4_1(Y))
        if self.debug: print("conv4_1:\t\t", Y.shape)
        
        Y = F.relu(self.conv4_2(Y))
        if self.debug: print("conv4_2:\t\t", Y.shape)

        Y = F.relu(self.conv4_3(Y))
        if self.debug: print("conv4_2:\t\t", Y.shape)
            
        Y = F.max_pool2d(Y, 2)
        if self.debug: print("max_pool4:\t", Y.shape)  
        
        # fc5
        Y = F.relu(self.fc5(Y))
        if self.debug: print("fc5:\t", Y.shape)
        Y = self.dropout5(Y)
        
        # fc6
        Y = F.relu(self.fc6(Y))
        if self.debug: print("fc6:\t", Y.shape)
        Y = self.dropout6(Y)

        # fc7
        Y = torch.sigmoid(self.fc7(Y))
        if self.debug: print("fc7:\t", Y.shape)
        
        # Remove unnecessary dimensions and change shape to [10,3]
        # to make predictions
        output = torch.squeeze(Y)
        if self.debug: print("output:\t", output.shape)
            
        return output

    def model_3(self, X):
        if self.debug: print("model3 input:\t", X.shape) 
            
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
            
        X = F.relu(self.fc6(X))
        if self.debug: print("fc6:\t\t", X.shape)
        
        X = torch.sigmoid(self.fc7(X))
        if self.debug: print("fc7:\t\t", X.shape)
            
        # Remove unnecessary dimensions and change shape to match target tensor
        # [10, num_cats, 1, 1] --> [10,num_cats] to make predicitons
        output = torch.squeeze(X)
        if self.debug: print("output:\t", output.shape)
            
        return output