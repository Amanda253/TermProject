from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
#from ConvNet import ConvNet
import numpy as np 
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from torchvision import transforms as transforms
import PIL
import itertools


class ConvNet(nn.Module):
    def __init__(self, mode=1, debug=False, num_cats=92):
        super(ConvNet, self).__init__()
        self.debug = debug

        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, assign one of the valid mode.
        # This will fix the forward function (and the network graph) for training/testing
        if mode == 1:
            # Uses square kernels of size 3x3 with stride 1
            # Include padding of size P = (K-1)/2 to preserve dimensions 
            '''
            self.conv1 = nn.Conv2d(3,16,5)
            self.conv2 = nn.Conv2d(16,32,5)
            self.conv3 = nn.Conv2d(32,64, 5)
            
            self.fc1 = nn.Linear(64*115*115, 32) 
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, num_cats)
            self.dropout = nn.Dropout(p=.5)
            '''
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
            
            self.dropout = nn.Dropout(p=.5)
            
            # Add dropout to randomly make some pixels 0 with the probability p
            # Use dropout2d since it is a spatial-dropout designed for 4-D tensors such as images or feature maps from convolution layers. The standard dropout will not be able to effectively regularize the network.

            
            # COCO images have possible 91 category ids to predict
            # Define fcNN which performs the classification
            self.fc1 = nn.Linear(128 * 4 * 4, 1024)
            self.fc2 = nn.Linear(1024, num_cats)
            self.forward = self.model_1

        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
    def model_1(self, X):
        if self.debug: print("model1 input:\t", X.shape)

        def num_flat_features(X):
            size = X.size()[1:]
            num_features = 1
            for s in size:
              num_features *= s
            return num_features 
        '''    
        X = self.conv1(X)
        X = self.conv2(X)
        X = F.max_pool2d(self.conv3(X), (2, 2), stride=1)
        #print(X.shape)
        X = X.view(-1, num_flat_features(X))
        X = self.fc1(X)
        #X = self.fc2(X)
        X = self.dropout(self.fc2(X))
        X = F.relu(self.fc3(X))
        m = nn.Sigmoid()
        X = m(X)
            
        return X
        '''
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



def train(model, device, train_loader, optimizer, criterion, epoch, batch_size, num_cats):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    accuracy = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # Compute loss based on criterion
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()

        # Get predicted class by rounding 
        pred = output.round()
        # Count correct predictions overall 
        # Get element-wise equality between the preds and the targets for this batch,
        # finally sum the equalities and convert to a python float
        num_equal = pred.eq(target).sum().item()
        correct += num_equal
        batch_accuracy = num_equal / torch.numel(target)
        accuracy += batch_accuracy
        
        progbar(batch_idx, len(train_loader), 10, batch_accuracy)
            
    train_loss = float(np.mean(losses))
    train_acc = 100. * correct / ((batch_idx+1) * batch_size * num_cats)
    print('Train set\t Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(np.mean(losses)), correct, (batch_idx+1) * batch_size * num_cats, train_acc))
          
    return train_loss, train_acc

def test(model, device, test_loader, criterion, num_cats):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    accuracy = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
                
            # Predict for data by doing forward pass
            output = model(data)

            # Compute loss based on same criterion as training
            loss = criterion(output, target)

            # Append loss to overall test loss
            losses.append(loss.item())

            # Get predicted class by rounding 
            pred = output.round()
            # Count correct predictions overall 
            # Get element-wise equality between the preds and the targets for this batch,
            # finally sum the equalities and convert to a std python float
            num_equal = pred.eq(target).sum().item()
            correct += num_equal
            accuracy += num_equal / torch.numel(target)

    test_loss = float(np.mean(losses))
    test_acc = (100. * correct) / (len(test_loader.dataset) * num_cats)
    print('\nTest set\t Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset) * num_cats, test_acc))
    
    return test_loss, test_acc

def list_flatten(lists):
    return [item for sublist in lists for item in sublist]

def equalize_dis(catnms, idList, coco_dataset):
    
    '''
    function that takes in coco image ids and returns
    an id list containg balanced image classes which
    may contain overlaping categories
    '''
    
    dis_list = []
    
    # smallest list of categories
    minSet = len(coco_dataset.coco.getImgIds(imgIds=idList, catIds=catnms[0]))
    
    # seperates the id list into seperate categories
    for i in catnms:
        temp = coco_dataset.coco.getImgIds(imgIds=idList, catIds=i)
        dis_list.append(temp)
        if len(temp) < minSet:
            minSet = len(temp)
            
    # displays the bar graph of the unbalanced 
    list_lengths = [len(x) for x in dis_list]
    plt.bar(range(len(catnms)), height=list_lengths)
    plt.xlabel("Number of classes")
    plt.ylabel("Number of images")
    plt.show()
    
    new_list = []

    # reduces the lists to the size of the smallest list
    for l in dis_list:
        new_list.append(l[slice(minSet)])
        print(len(l))
    
    # displays the balanced list
    list_lengths = [len(x) for x in new_list]
    plt.bar(range(len(catnms)), height=list_lengths)
    plt.xlabel("Number of classes")
    plt.ylabel("Number of images")
    plt.show()
    
    # combining all the elements into a single list    
    single_list = list(itertools.chain(*new_list))
    
    return single_list 

def removeList(l1, l2):
    return [i for i in l1 if i not in l2]

def equalize_dis_no_overlap(catnms, idList, coco_dataset):
    
    '''
    function that takes in coco image ids and returns
    an id list containg balanced image classes with
    no overlaping categories
    '''
    
    dis_list = []
    
    # get list of imgIds that correspond to the given categories
    for i in catnms:
        temp = coco_dataset.coco.getImgIds(imgIds=idList, catIds=i)
        dis_list.append(temp)
    
    # sort list of lists from least to greatest
    dis_list.sort(key=len)

    # Displaying the class distribution before
    list_lengths = [len(x) for x in dis_list]
    plt.bar(range(len(catnms)), height=list_lengths)
    plt.xlabel("Number of classes")
    plt.ylabel("Number of images")
    plt.show()
    
    # a list for values to remove and a list for the new values
    remove_list = []
    new_list = []
    
    # The minimum length it will be reduced to
    min_length = len(dis_list[0])
    
    # Getting rid of overlaping categories in the
    # category list
    for li in dis_list:
        
        r_list = removeList(li, remove_list)
        
        new_list.append(r_list)
        remove_list.extend(r_list)
        
        if len(r_list) < min_length:
            min_length = len(r_list)
            
    final_list = []

    # turning the category lists into a single list
    for l in new_list:
        final_list.append(l[slice(min_length)])
        
    # Displaying the balanced class distribution
    list_lengths = [len(x) for x in final_list]
    plt.bar(range(len(catnms)), height=list_lengths)
    plt.xlabel("Number of classes")
    plt.ylabel("Number of images")
    plt.show()
     
    # combining all the elements into a single list 
    single_list = list(itertools.chain(*final_list))
    
    
    return single_list 


def coco_subset(coco_dataset, catIds):
    # get the corresponding image ids containing the categories 
    imgIds = []
    for L in range(0, len(catIds)+1):
        for subset in itertools.combinations(catIds, L):
            if len(subset) > 0:
                imgIds.append(coco_dataset.coco.getImgIds(catIds=list(subset)))
            
    imgIds = np.unique(np.array(list_flatten(imgIds)))
    
    imgIds = equalize_dis_no_overlap(catIds, imgIds, coco_dataset)
    
    # convert the coco image ids to numpy array
    ids = np.array(coco_dataset.ids)
    # locate indicies of corresponding images ids as 1D array
    idxs = np.array(list(map(lambda x: np.where(ids == x), imgIds))).flatten()
    # select subset of intrest from the full coco dataset with these indicies   
    subset = torch.utils.data.Subset(coco_dataset, idxs)
    return subset

def k_hot_catIDs(batch, catIds):
    # There are 80 classes total in COCO, but catIds go up to 91
    imgs = []
    target = torch.zeros(len(batch), len(catIds))
    for i, (img, segs) in enumerate(batch):
        imgs.append(img)
        img_catIds = [seg['category_id'] for seg in segs]
        for k, catId in enumerate(catIds):
            target[i,k] = int(catId in img_catIds)
    
    imgs = torch.stack(imgs)      
    return imgs, target

def progbar(curr, total, full_progbar, accuracy):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', 
          '#'*filled_progbar + '-'*(full_progbar-filled_progbar), 
          '[{:>7.2%}]'.format(frac), 
          'Accuracy: [{:>7.2%}]'.format(accuracy), 
          end='')

def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    debug = FLAGS.debug

    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    root = 'D:/Dataset/'
    train_root = 'train2017'
    train_path = '{}/{}'.format(root, train_root)
    train_annFile = '{}annotations/instances_{}.json'.format(root, train_root)
    test_root = 'val2017'
    #test_path = '{}{}'.format(root, test_root)
    test_path = 'D:/Dataset/val2017/val2017'
    test_annFile = '{}annotations/instances_{}.json'.format(root, test_root)

    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    print('Loading training data...')
    train_dataset = datasets.CocoDetection(root = train_path,
                                      annFile = train_annFile,
                                      transform=transform)
    print('Loading validataion data...')
    test_dataset = datasets.CocoDetection(root = test_path,
                                      annFile = test_annFile,
                                      transform=transform)

    # Create dataloaders for cull coco dataset
#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, 
#                             num_workers=0, collate_fn=k_hot_catIDs)
#     test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, 
#                             num_workers=0, collate_fn=k_hot_catIDs)
    # define categories of interest
    catNms=['bird','bench','handbag']
    # get the category ids of interest
    catIds = train_dataset.coco.getCatIds(catNms=catNms)
    train_catIds = train_dataset.coco.getCatIds(catNms=catNms)
    test_catIds = test_dataset.coco.getCatIds(catNms=catNms)
    # Create subsets of coco dataset using selected categories      
    train_subset = coco_subset(train_dataset, train_catIds)
    test_subset = coco_subset(test_dataset, test_catIds)
    # Prepare data loaders for these subsets
    collate_fn = lambda b: k_hot_catIDs(b, catIds)
    train_collate_fn = lambda b: k_hot_catIDs(b, train_catIds)
    test_collate_fn = lambda b: k_hot_catIDs(b, test_catIds)
    train_subset_loader = DataLoader(train_subset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_subset_loader = DataLoader(test_subset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
    
    
    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode, debug, len(catIds)).to(device)
    print('mode {}'.format(FLAGS.mode))
    print(model)

    # Use Binary Cross Entropy as the loss function 
    # since we want to allow multiple lables for each input. 
    # Cross Entropy will map the networks predictions to a probabilities in range [0,1] 
    criterion = nn.BCELoss()
    
    # Define optimizer function.
    # Use stochastic gradient descent optimizer with momentum
    # and a decaying a learning rate 
    learning_rate = FLAGS.learning_rate
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
    
    # Define tracked network performance metrics
    best_accuracy = 0.0
    train_losses = np.zeros(FLAGS.num_epochs)
    train_accuracies = np.zeros(FLAGS.num_epochs)
    test_losses = np.zeros(FLAGS.num_epochs)
    test_accuracies = np.zeros(FLAGS.num_epochs)
    
    # Define name of this model for bookkeeping
    name = 'model{}_lr{}_epochs{}_{}'.format(FLAGS.mode, FLAGS.learning_rate, FLAGS.num_epochs, FLAGS.name)
    # Define the tensorboard writer to track netowrk training
    writer = SummaryWriter('./runs/COCO2017/{}'.format(name))
    # Define the output filename for the training and evaluation traces
    filename = "./output/COCO2017_{}.dat".format(name)    
    num_cats = len(catNms)
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        print("Epoch {}".format(epoch))
        train_loss, train_accuracy = train(model, device, 
                                           train_subset_loader, 
                                           optimizer, criterion, 
                                           epoch, FLAGS.batch_size, 
                                           num_cats)
        test_loss, test_accuracy = test(model, device, test_subset_loader, criterion, num_cats)
        
        # Store epoch metrics in memory
        i = epoch - 1
        train_losses[i] = train_loss
        train_accuracies[i] = train_accuracy
        test_losses[i] = test_loss
        test_accuracies[i] = test_accuracy 
            
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
    
        # Log the epoch metrics in tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)  
        
    # Flush all writer logs and close resource
    writer.flush()
    writer.close()
    
    with open(filename, 'w') as f:
        f.write('Train Loss,Train Accuracy,Test Loss, Test Accuracy')
        for epoch in range(FLAGS.num_epochs):
            # Write epoch metrics to disk
            f.write('{},{},{},{}\n'.format(train_losses[epoch],
                                         train_accuracies[epoch],
                                         test_losses[epoch],
                                         test_accuracies[epoch]))

    # Print final results to console
    print("best accuracy was {:2.2f}".format(best_accuracy))
    print("Training and evaluation finished")
    
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode', type=int, default=1, help='Select mode between 1-5.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to put logging.')
    parser.add_argument('--debug', type=bool, default=False, help='Enable debug mode.')
    parser.add_argument('--name', type=str, default='model', help='Set model name')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)