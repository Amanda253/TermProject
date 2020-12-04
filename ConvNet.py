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
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

            self.conv2 = nn.Conv2d(in_channels=32,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            self.conv3 = nn.Conv2d(in_channels=64,
                                   out_channels=128,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            # Add dropout to randomly make some pixels 0 with the probability p
            # Use dropout2d since it is a spatial-dropout designed for 4-D tensors such as images or feature maps from convolution layers. The standard dropout will not be able to effectively regularize the network.
            self.dropout = nn.Dropout2d(p=0.5)
            
            # CIFAR-10 images are of objects in 10 classes
            # Define fcNN which performs the classification
            D_in = 128 * 4 * 4
            self.fc1 = nn.Linear(D_in, 1024)
            self.fc2 = nn.Linear(1024, 10)
            self.forward = self.model_1

        elif mode == 2:            
            self.conv1 = nn.Conv2d(in_channels=3,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

            self.conv2 = nn.Conv2d(in_channels=32,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            self.conv3 = nn.Conv2d(in_channels=64,
                                   out_channels=128,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            self.conv4 = nn.Conv2d(in_channels=128,
                                   out_channels=256,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            # Add dropout to randomly make some pixels 0 with the probability p
            # Use dropout2d since it is a spatial-dropout designed for 4-D tensors such as images or feature maps from convolution layers. The standard dropout will not be able to effectively regularize the network.
            self.dropout = nn.Dropout2d(p=0.5)
            
            # CIFAR-10 images are of objects in 10 classes
            # Define fcNN which performs the classification
            D_in = 256 * 2 * 2
            self.fc1 = nn.Linear(D_in, 1024)
            self.fc2 = nn.Linear(1024, 10)
            self.forward = self.model_2
        elif mode == 3:
            # Uses square kernels of size 3x3 with stride 1
            # Include padding of size P = (K-1)/2 to preserve dimensions 
            self.conv1 = nn.Conv2d(in_channels=3,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)


            self.conv2 = nn.Conv2d(in_channels=32,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            

            self.conv3 = nn.Conv2d(in_channels=64,
                                   out_channels=128,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            self.conv4 = nn.Conv2d(in_channels=128,
                                   out_channels=256,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            # Add dropout to randomly make some pixels 0 with the probability p
            # Use dropout2d since it is a spatial-dropout designed for 4-D tensors such as images or feature maps from convolution layers. The standard dropout will not be able to effectively regularize the network.
            self.dropout = nn.Dropout2d(p=0.5)
            
            # CIFAR-10 images are of objects in 10 classes
            # Define fcNN which performs the classification
            D_in = 256 * 4 * 4
            self.fc1 = nn.Linear(D_in, 1024)
            self.fc2 = nn.Linear(1024, 10)
            self.forward = self.model_3
        elif mode == 4:
            # Include padding of size P = (K-1)/2 to preserve dimensions 
            self.conv1 = nn.Conv2d(in_channels=3,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

            self.conv2 = nn.Conv2d(in_channels=32,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            self.conv3 = nn.Conv2d(in_channels=64,
                                   out_channels=128,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            self.conv4 = nn.Conv2d(in_channels=128,
                                   out_channels=256,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            self.conv5 = nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            # Add dropout to randomly make some pixels 0 with the probability p
            # Use dropout2d since it is a spatial-dropout designed for 4-D tensors such as images or feature maps from convolution layers. The standard dropout will not be able to effectively regularize the network.
            self.dropout = nn.Dropout2d(p=0.5)
            
            # CIFAR-10 images are of objects in 10 classes
            # Define fcNN which performs the classification
            D_in = 512 * 4 * 4
            self.fc1 = nn.Linear(D_in, 1024)
            self.fc2 = nn.Linear(1024, 10)
            self.forward = self.model_4
        elif mode == 5:
            # Uses square kernels of size 3x3 with stride 1
            # Include padding of size P = (K-1)/2 to preserve dimensions 
            self.conv1 = nn.Conv2d(in_channels=3,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

            self.conv2 = nn.Conv2d(in_channels=32,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            self.conv3 = nn.Conv2d(in_channels=64,
                                   out_channels=128,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            self.conv4 = nn.Conv2d(in_channels=128,
                                   out_channels=256,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            self.conv5 = nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            self.conv6 = nn.Conv2d(in_channels=512,
                                   out_channels=1024,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
            
            # Add dropout to randomly make some pixels 0 with the probability p
            # Use dropout2d since it is a spatial-dropout designed for 4-D tensors such as images or feature maps from convolution layers. The standard dropout will not be able to effectively regularize the network.
            self.dropout = nn.Dropout2d(p=0.5)
            
            # CIFAR-10 images are of objects in 10 classes
            # Define fcNN which performs the classification
            D_in = 1024 * 4 * 4
            self.fc1 = nn.Linear(D_in, 1024)
            self.fc2 = nn.Linear(1024, 10)
            self.forward = self.model_5
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
            
        X = self.dropout(X)
        if self.debug: print("dropout:\t", X.shape)
        
        X = torch.flatten(X, start_dim=1)
        if self.debug: print("flatten:\t", X.shape)
            
        X = F.relu(self.fc1(X))
        if self.debug: print("fc1:\t\t", X.shape)
        
        output = nn.Sigmoid(self.fc2(X))
        if self.debug: print("fc2:\t\t", X.shape)
            
        return output

    def model_2(self, X):
        if self.debug: print("model2 input:\t", X.shape) 
            
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
            
        X = self.dropout(X)
        if self.debug: print("dropout:\t", X.shape)
        
        X = torch.flatten(X, start_dim=1)
        if self.debug: print("flatten:\t", X.shape)
            
        X = F.relu(self.fc1(X))
        if self.debug: print("fc1:\t\t", X.shape)
        
        output = self.fc2(X)
        if self.debug: print("fc2:\t\t", X.shape)
            
        return output
    
    def model_3(self, X):
        if self.debug: print("model3 input:\t", X.shape) 
            
        X = F.relu(self.conv1(X))
        if self.debug: print("conv1:\t\t", X.shape)
        
        X = F.relu(self.conv2(X))
        if self.debug: print("conv2:\t\t", X.shape)
            
        X = F.max_pool2d(X, 2)
        if self.debug: print("max_pool1:\t", X.shape)
        
        X = F.relu(self.conv3(X))
        if self.debug: print("conv3:\t\t", X.shape)
        
        X = F.max_pool2d(X, 2)
        if self.debug: print("max_pool2:\t", X.shape)
        
        X = F.relu(self.conv4(X))
        if self.debug: print("conv3:\t\t", X.shape)
        
        X = F.max_pool2d(X, 2)
        if self.debug: print("max_pool3:\t", X.shape)
            
        X = self.dropout(X)
        if self.debug: print("dropout:\t", X.shape)
        
        X = torch.flatten(X, start_dim=1)
        if self.debug: print("flatten:\t", X.shape)
            
        X = F.relu(self.fc1(X))
        if self.debug: print("fc1:\t\t", X.shape)
        
        output = self.fc2(X)
        if self.debug: print("fc2:\t\t", X.shape)
            
        return output
    
    def model_4(self, X):
        if self.debug: print("model4 input:\t", X.shape) 
            
        X = F.relu(self.conv1(X))
        if self.debug: print("conv1:\t\t", X.shape)
        
        X = F.relu(self.conv2(X))
        if self.debug: print("conv2:\t\t", X.shape)
            
        X = F.relu(self.conv3(X))
        if self.debug: print("conv3:\t\t", X.shape)
            
        X = F.max_pool2d(X, 2)
        if self.debug: print("max_pool1:\t", X.shape)

        X = F.relu(self.conv4(X))
        if self.debug: print("conv3:\t\t", X.shape)
        
        X = F.max_pool2d(X, 2)
        if self.debug: print("max_pool2:\t", X.shape)
 
        X = F.relu(self.conv5(X))
        if self.debug: print("conv3:\t\t", X.shape)
            
        X = F.max_pool2d(X, 2)
        if self.debug: print("max_pool3:\t", X.shape)
            
        X = self.dropout(X)
        if self.debug: print("dropout:\t", X.shape)
        
        X = torch.flatten(X, start_dim=1)
        if self.debug: print("flatten:\t", X.shape)
            
        X = F.relu(self.fc1(X))
        if self.debug: print("fc1:\t\t", X.shape)
        
        output = self.fc2(X)
        if self.debug: print("fc2:\t\t", X.shape)
            
        return output
    
    def model_5(self, X):
        if self.debug: print("model1 input:\t", X.shape) 
            
        X = F.relu(self.conv1(X))
        if self.debug: print("conv1:\t\t", X.shape)
        
        X = F.relu(self.conv2(X))
        if self.debug: print("conv2:\t\t", X.shape)
        
        X = F.max_pool2d(X, 2)
        if self.debug: print("max_pool1:\t", X.shape)
        
        X = F.relu(self.conv3(X))
        if self.debug: print("conv3:\t\t", X.shape)
                
        X = F.relu(self.conv4(X))
        if self.debug: print("conv4:\t\t", X.shape)
        
        X = F.max_pool2d(X, 2)
        if self.debug: print("max_pool2:\t", X.shape)  
            
        X = F.relu(self.conv5(X))
        if self.debug: print("conv3:\t\t", X.shape)
                
        X = F.relu(self.conv6(X))
        if self.debug: print("conv4:\t\t", X.shape)
        
        X = F.max_pool2d(X, 2)
        if self.debug: print("max_pool2:\t", X.shape)  
            
        X = self.dropout(X)
        if self.debug: print("dropout:\t", X.shape)
        
        X = torch.flatten(X, start_dim=1)
        if self.debug: print("flatten:\t", X.shape)
            
        X = F.relu(self.fc1(X))
        if self.debug: print("fc1:\t\t", X.shape)
        
        output = self.fc2(X)
        if self.debug: print("fc2:\t\t", X.shape)
            
        return output