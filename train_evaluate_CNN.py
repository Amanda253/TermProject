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
from ConvNet import ConvNet
import numpy as np 

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
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
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        max_val, preds = torch.max(output, dim=1)
        # Count correct predictions overall 
        # Reshape the target tensor to match the shape of the preds, then
        # Get element-wise equality between the preds and the targets for this batch,
        # finally sum the equalities and convert to a std python float
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    train_loss = float(np.mean(losses))
    train_acc = 100. * correct / ((batch_idx+1) * batch_size)
    
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(np.mean(losses)), correct, (batch_idx+1) * batch_size, train_acc))
          
    return train_loss, train_acc

def test(model, device, test_loader, criterion):
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

            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            max_val, preds = torch.max(output, dim=1)
            # Count correct predictions overall 
            # Reshape the target tensor to match the shape of the preds, then
            # Get element-wise equality between the preds and the targets for this batch,
            # finally sum the equalities and convert to a std python float
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = float(np.mean(losses))
    accuracy = 100. * correct / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    
    return test_loss, accuracy

def k_hot_catIDs(batch):
    # print('length: {}'.format(len(batch)))
    out = []
    for data in batch:
        img, segs = data
        cats = [seg['category_id'] for seg in segs]
        out.append((img, cats))
    return out

def coco_subset_dataloader(coco_dataset, catNms, batch_size):
    # get the category ids of interest
    catIds = coco_dataset.coco.getCatIds(catNms=catNms)
    # get the corresponding image ids containing the categories 
    imgIds = coco_dataset.coco.getImgIds(catIds=catIds)
    # convert the coco image ids to numpy array
    ids = np.array(coco_dataset.ids)
    # locate indicies of corresponding images ids as 1D array
    idxs = np.array(list(map(lambda x: np.where(ids == x), imgIds))).flatten()
    # select subset of intrest from the full coco dataset with these indicies   
    subset = torch.utils.data.Subset(coco_dataset, idxs)
    # Prepare a data loader for this subset dataset
    subset_loader = DataLoader(subset, 
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=0,
                               collate_fn=k_hot_catIDs)
    return subset_loader

def run_main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    debug = FLAGS.debug

    # Initialize the model and send to device 
    model = ConvNet(FLAGS.mode, debug).to(device)
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
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.9, weight_decay=5e-4)

    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    root = 'F:/Research/datasets/MSCOCO2014'
    train_root = 'train2014'
    train_path = '{}/train2014'.format(root, train_root)
    train_annFile = '{}/annotations/instances_{}.json'.format(root, train_root)
    test_root = 'train2014'
    test_path = '{}/train2014'.format(root, test_root)
    test_annFile = '{}/annotations/instances_{}.json'.format(root, test_root)

    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    train_dataset = datasets.CocoDetection(root = train_root,
                                      annFile = train_annFile,
                                      train=True, 
                                      transform=transform)
    test_dataset = datasets.CocoDetection(root = test_path,
                                      annFile = test_annFile,
                                      train=False,
                                      transform=transform)
    
    # Create dataloaders for cull coco dataset
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, 
                            num_workers=0, collate_fn=select_catIDs)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, 
                            num_workers=0, collate_fn=select_catIDs)
    
    # Create dataloaders for coco dataset of select categories 
    catNms=['person','bench','handbag']
    train_subset_loader = coco_subset_dataloader(train_dataset, catNms, FLAGS.batch_size)
    test_subset_loader = coco_subset_dataloader(test_dataset, catNms, FLAGS.batch_size)
    
    # Define tracked network performance metrics
    best_accuracy = 0.0
    train_losses = np.zeros(FLAGS.num_epochs)
    train_accuracies = np.zeros(FLAGS.num_epochs)
    test_losses = np.zeros(FLAGS.num_epochs)
    test_accuracies = np.zeros(FLAGS.num_epochs)
    
    # Define the tensorboard writer to track netowrk training
    writer = SummaryWriter('./runs/COCO2014/Dropout/{}'.format(FLAGS.name))
    
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        print("Epoch {}".format(epoch))
        train_loss, train_accuracy = train(model, device, train_loader,
                                            optimizer, criterion, epoch, FLAGS.batch_size)
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        
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

    # Print final results to console
    print("accuracy is {:2.2f}".format(best_accuracy))
    print("Training and evaluation finished")
    
    
if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=30,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=16,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    parser.add_argument('--debug',
                        type=bool,
                        default=False,
                        help='Enable debug mode.')
    parser.add_argument('--name',
                        type=str,
                        default='model',
                        help='Set model name')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
    
    