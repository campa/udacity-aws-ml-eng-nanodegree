#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import argparse
import os
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer):    
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
    
    #freeze model params
    for param in model.parameters():
        param = param.requires_grad_(False)

    #new layer
    model.fc = nn.Sequential(
                        nn.Linear(model.fc.in_features, 256),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(256, 133),               # num of classes    ? 
                        nn.LogSoftmax(dim=1))


    logger.info("The new layer is : %s ",model.fc)
    # model = model.to(device) #Moving the model to GPU
    
    return model

def _get_test_data_loader(batch_size, training_dir):
    logger.info("Get test data loader")
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset_test = datasets.ImageFolder(training_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)
    return test_loader

def _get_train_data_loader(batch_size, training_dir):
    logger.info("Get train data loader")
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset_train = datasets.ImageFolder(training_dir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    return train_loader

def save_model(model, model_dir):
    '''
    TODO: Save the trained model
    '''
    # torch.save(model, path)
    logger.info("Saving the model.")
    # path = os.path.join(args.model-dir, "model.pth")
    # torch.save(model.cpu().state_dict(), path)

    torch.save(model, model-dir)
    # torch.save(model, "nlp-img-classification-train-deploy.pt")

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    # model=train(model, train_loader, loss_criterion, optimizer)
   
    # train_kwargs = {"batch_size": args.batch_size}

    train_loader = _get_train_data_loader(args.batch_size, args.train)
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    # test(model, test_loader, criterion)

    test_loader = _get_train_data_loader(args.batch_size, args.test)
    test(model, test_loader, loss_criterion)
    
    save_model(model, args.model_dir)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''

     # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    args = parser.parse_args()
    logger.info("hpo.args: " + args)
    main(args)
