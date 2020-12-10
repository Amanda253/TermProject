# Fully Convolutional Neural Network for Multiple-Label Object Recognition in Images

This is an implimentation of FCN for multiple-label object recognition using K-hot vector enocding. This model is trained and evaluated with the MS COCO2017 dataset.
The follow python libraries are required:

-   PyTorch
-   Numpy
-   Matplotlib
-   Argparse
-   OS
-   Time
-   Itertools

You need to download the MS COCO 2017 train/val dataset and annotations seperatly. Then update the path strings inside the run_main function of train_evaluate_CNN.py to point to the location of the COCO datasets.

## How to train and evaluate models

-   To start training a model, open two terminal instances and nativate to the project folder.
-   With the project folder as the working directory start tensorboard to view model output
    -   Enter "tensorboard --logdir runs" command in one of the terminal intances
    -   Open a browser window and navigate to the indicated localhost url
-   Start model training and evaluation by entering command:
    -   "python .\train_evaluate_CNN.py"
    -   This script will accept the following arguments
        -   --mode
            -   Select which model to evaluation.
            -   Valid options: 1-3
            -   default: 3 (best)
        -   --learning_rate
            -   Set the learning rate for this evaluation.
            -   default: 0.01 (best)
        -   --num_epochs
            -   Set the number of epochs to train over
            -   default: 15
        -   --batch_size
            -   Set the batch size for each epoch. Best if this divides the dataset size evenly
            -   default: 10
        -   --debug
            -   enable/disable debug mode which will print all model layer sizes when the model's forward function is called
            -   default: False
        -   --name
            -   Set a unive name for the model
            -   default: ''
        -   --balance_dataset
            -   enable/disable MS COCO2017 subset balancing
            -   default: False
        -   --num_train_samples
            -   Number of samples of the dataset subset used to train the model.
            -   default: 0 (use full subset)
        -   --num_test_samples
            -   Number of samples of the dataset subset used to test the model.
            -   default: 0 (use full subset)
        -   --num_cats
            -   Number of categories used to evaluate the model.
            -   Valid options: [1-8]
            -   default: 3
-   Refreshing the Tensorboard tab will update the performance graphs with data from the most recent epoch.
-   All performacne data is also saved to an output .dat file located in the output directory.

## Github Link

https://github.com/Amanda253/TermProject
