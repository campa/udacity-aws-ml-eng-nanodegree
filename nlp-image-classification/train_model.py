#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

import smdebug.pytorch as smd


#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

    

def train(model, train_loader, optimizer, epoch, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    hook.set_mode(smd.modes.TRAIN)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

    
    
def net(model-dir):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''

    model = torch.load(model-dir)
    return model

def _get_test_data_loader(test_batch_size, training_dir):
    logger.info("Get test data loader")
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset_test = datasets.ImageFolder(training_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch_size)
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

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    # model=net()
    model=net(args.model-dir)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader = _get_train_data_loader(args.batch_size, args.train)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test_loader = _get_train_data_loader(args.test_batch_size, args.test)
    
    '''
    TODO: Save the trained model
    '''

    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, epoch, hook)
        test(model, test_loader, hook)

    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
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
    print("hpo.args", args)
    main(args)
