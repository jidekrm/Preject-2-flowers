import argparse
import torch
from torch import nn
from torch import optim
from torchvision import models, transforms, datasets


'''  
define models. Densenet121 and Densenet161 are used here because of my limited GPU power. 
Other models failed cause of low memory
''' 

densenet161 = models.densenet161(pretrained = True)
densenet121 = models.densenet121(pretrained = True)
models = {'densenet121': densenet121, 'densenet161': densenet161}

def create_model(model_name, learning_rate, hidden_units):
    """
    Create a model with the given model name, learning rate, and number of hidden units.

    Parameters:
        model_name (str): The name of the model to be created.
        learning_rate (float): The learning rate for the optimizer.
        hidden_units (int): The number of hidden units for the model's classifier.

    Returns:
        model (torch.nn.Module): The created model.
        criterion (torch.nn.Module): The loss criterion for the model.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
    """
    model = models[model_name]

    for param in model.parameters():
        param.requires_grad = False


    from collections import OrderedDict
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2208, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ])
    )
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= learning_rate)
    torch.cuda.set_device(1) 
    return model, criterion, optimizer




def train_model(model, trainloader, testloader, criterion, optimizer, epochs,  device):
    """
    Trains a machine learning model using the given trainloader for the specified number of epochs.
    
    Args:
        model (nn.Module): The model to be trained.
        trainloader (DataLoader): The data loader for training data.
        testloader (DataLoader): The data loader for test/validation data.
        criterion (loss function): The loss function used to calculate the loss.
        optimizer (optimizer): The optimizer used to update the model parameters.
        epochs (int): The number of epochs to train the model.
        device (torch.device): The device on which the model and data should be loaded.
    
    Returns:
        None
    """
    
    model.to(device)
    print_every = 5
    step = 0
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            step += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % print_every == 0:
                validation(model, criterion, epochs, device, testloader, print_every, epoch, running_loss)   
                running_loss = 0
                model.train()
    print('Training Done!')




def validation(model, criterion, epochs, device, testloader, print_every, epoch, running_loss):
    """
    Performs validation on a given model.

    Args:
        model (torch.nn.Module): The model to be validated.
        criterion (torch.nn.Module): The loss function used for validation.
        epochs (int): The total number of epochs for validation.
        device (torch.device): The device on which the model is trained.
        testloader (torch.utils.data.DataLoader): The data loader for the validation dataset.
        print_every (int): The number of iterations after which to print the validation statistics.
        epoch (int): The current epoch number.
        running_loss (float): The running loss value for the training dataset.

    Returns:
        None
    """
    
    test_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
        
        
            



def save_checkpoint(model, checkpoint_save_path, train_data, optimizer, epochs, model_name):
    """
    Save the model checkpoint to the specified path.

    Parameters:
        - model: The model to save the checkpoint for.
        - checkpoint_save_path: The path where the checkpoint should be saved.
        - train_data: The training data used by the model.
        - optimizer: The optimizer used for training the model.
        - epochs: The number of epochs the model was trained for.

    Returns:
        None
    """
    
    model.class_to_idx = train_data.class_to_idx
    input_size = model.classifier.fc1.in_features
    output_size = model.classifier.fc2.out_features
    drop_out = model.classifier.dropout.p
    hidden_layer_size = model.classifier.fc1.in_features


    checkppoint = {'input_size': input_size,
             'output_size': output_size,
             'hidden_layer_size':hidden_layer_size,
             'drop_out': drop_out,
             'optimizer' : optimizer.state_dict(),
              'epochs': epochs,
              'model_name': model_name,
              'class_to_idx': model.class_to_idx,
              'classifier': model.classifier,
             'state_dict': model.state_dict()}
             

    torch.save(checkppoint, checkpoint_save_path)


def create_transforms():
    """
    Create and return three different image transforms for training, testing, and validation.

    Returns:
        train_transform (torchvision.transforms.Compose): A composition of image transformations for training.
        test_transform (torchvision.transforms.Compose): A composition of image transformations for testing.
        valid_transform (torchvision.transforms.Compose): A composition of image transformations for validation.
    """
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize

    ])

    valid_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    return train_transform, test_transform, valid_transform


def create_data(train_dir, test_dir, valid_dir):
    """
    Creates data for training, testing, and validation.

    Parameters:
        train_dir (str): The directory containing the training data.
        test_dir (str): The directory containing the testing data.
        valid_dir (str): The directory containing the validation data.

    Returns:
        train_data (torchvision.datasets.ImageFolder): The training data.
        test_data (torchvision.datasets.ImageFolder): The testing data.
        valid_data (torchvision.datasets.ImageFolder): The validation data.
    """
    
    train_transform, test_transform, valid_transform = create_transforms()
    train_data = datasets.ImageFolder(train_dir, transform = train_transform)
    test_data = datasets.ImageFolder(test_dir, transform = test_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transform)
    return train_data, test_data, valid_data



def create_data_loaders( train_data, test_data, valid_data, batch_size = 64):
    """
    Create data loaders for the given train, test, and valid datasets.

    Parameters:
        train_data (torch.utils.data.Dataset): The training dataset.
        test_data (torch.utils.data.Dataset): The testing dataset.
        valid_data (torch.utils.data.Dataset): The validation dataset.
        batch_size (int, optional): The batch size for the data loaders. Defaults to 64.

    Returns:
        torch.utils.data.DataLoader: The data loader for the training dataset.
        torch.utils.data.DataLoader: The data loader for the testing dataset.
        torch.utils.data.DataLoader: The data loader for the validation dataset.
    """
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    return trainloader,testloader,validloader




def get_args():
    """
    Parses command line arguments and returns an object containing the parsed arguments.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='flowers/train')
    parser.add_argument('--test_dir', type=str, default='flowers/test')
    parser.add_argument('--valid_dir', type=str, default='flowers/valid')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth')

    parser.add_argument('--model_name', type=str, default='densenet161')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--hidden_units', type=int, default=500)
    parser.add_argument('--device_name', type=str, default='gpu')

    in_args = parser.parse_args()
    return in_args

def arg_return(in_args):
    """
    Generates the function comment for the given function.

    Parameters:
        in_args (object): The input arguments object.

    Returns:
        tuple: A tuple containing the train_dir, test_dir, valid_dir, check_point, model_name, epochs, learning_rate, hidden_units, and device_name.
    """

    
    train_dir = in_args.train_dir
    test_dir = in_args.test_dir
    valid_dir = in_args.valid_dir
    check_point = in_args.checkpoint
    model_name = in_args.model_name
    epochs = in_args.epochs
    learning_rate = in_args.learning_rate
    hidden_units = in_args.hidden_units
    device_name = in_args.device_name
    return train_dir, test_dir, valid_dir, check_point, model_name, epochs, learning_rate, hidden_units, device_name


def check_command_line_args(in_args):
    """
    Prints the command line arguments passed to the function.

    Args:
        in_args: The command line arguments passed to the function.

    Returns:
        None
    """


    
    if in_args is None:
        print ('No command line arguments')
    else:
        print('Command Line Arguments:\n    train_dir = ', in_args.train_dir, '\n    test_dir = ',
               in_args.test_dir, '\n    valid_dir = ', in_args.valid_dir, '\n    checkpoint = ', in_args.checkpoint)




def main():
    """
    Runs the main function of the program.

    This function performs the following steps:
    1. Retrieves command line arguments using `get_args()`.
    2. Checks the validity of the command line arguments using `check_command_line_args()`.
    3. Extracts the required arguments from the command line arguments using `arg_return()`.
    4. Creates and loads the training, testing, and validation data using `create_data()` and `create_data_loaders()`.
    5. Creates the model, criterion, and optimizer using `create_model()`.
    6. Sets the device to be used for training based on the `device_name` argument.
    7. Trains the model using `train_model()`.
    8. Saves the trained model checkpoint using `save_checkpoint()`.

    Parameters:
    None

    Returns:
    None
    """

    
    in_args = get_args()
    check_command_line_args(in_args)
    train_dir, test_dir, valid_dir, checkpoint, model_name, epochs, learning_rate, hidden_units, device_name = arg_return(in_args)
 
    train_data, test_data, valid_data =create_data(train_dir, test_dir, valid_dir)
    trainloader, testloader, validloader = create_data_loaders(train_data, test_data, valid_data)
    model, criterion, optimizer = create_model(model_name= model_name, learning_rate=learning_rate, hidden_units=hidden_units)

    device = torch.device("cpu")
    if device_name == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    train_model(model= model,trainloader= trainloader, testloader= testloader, criterion = criterion , optimizer=optimizer , epochs=epochs, device=device)

    save_checkpoint(model=model, checkpoint_save_path=checkpoint, train_data=train_data,  optimizer=optimizer, epochs= epochs, model_name=model_name)


# Call to main function to run the program
if __name__ == "__main__":
    main()