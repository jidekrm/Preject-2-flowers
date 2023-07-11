import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models
import json



def process_image(image_path):
    """
    Processes an image for prediction by the model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The processed image as a NumPy array.
    """
    # get image and prepare it for prediction model

    pil_image = Image.open(image_path)
    pil_image = pil_image.resize((256, 256))
    width, height = pil_image.size
    left = (width - 224) // 2
    top = (height - 224) // 2
    right = (width + 224) // 2
    bottom = (height + 224) // 2
    pil_image = pil_image.crop((left, top, right, bottom))
    np_image = np.array(pil_image)/255
    np_image = (np_image -  [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
    np_image = np_image.transpose((2, 0, 1))
    return np_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""               
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
#     image = image.numpy().transpose((1, 2, 0))
    image = image.transpose((1, 2, 0)) #Image allready numpy array
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

'''  
define models. Densenet121 and Densenet161 are used here because of my limited GPU power. 
Other models failed cause of low memory
'''   
densenet161 = models.densenet161(pretrained = True)
densenet121 = models.densenet121(pretrained = True)
models = {'densenet121': densenet121, 'densenet161': densenet161}

def load_checkpoint(checkpoint_filepath):
    """
    Load a checkpoint from a file and return the corresponding model.
    
    Parameters:
    - `checkpoint_filepath` (str): The filepath of the checkpoint file to load.
    
    Returns:
    - `model` (model): The loaded model.
    """
    
    checkpoint = torch.load(checkpoint_filepath)
    
    model_name = checkpoint['model_name']
    model = models[model_name]

    epochs = checkpoint['epochs']
    classifier = checkpoint['classifier']
    optimizer = checkpoint['optimizer']
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model


# Laod mappings
def load_mapping(filepath):
    """
    Load a mapping from a file.

    Args:
        filepath (str): The path to the file containing the mapping.

    Returns:
        dict: The loaded mapping.
    """
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

    



def predict(image_path, model, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    torch.cuda.set_device(1)
    test_image = process_image(image_path)
    test_image = torch.from_numpy(test_image)
    test_image.unsqueeze_(0)
    test_image = test_image.float()

    model.to(device)
    model.eval()
    with torch.no_grad(): 
        
        logps = model.forward(test_image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk)
        return top_p[0].tolist(), top_class[0].tolist()
    


def print_top5(image_path,checkpoint):
    """
    Prints the top 5 predicted categories for a given image using a pre-trained model.

    Args:
        image_path (str): The path to the image file.
        checkpoint (str): The path to the saved model checkpoint.

    Returns:
        category_mapping (list): A list of the top 5 predicted categories for the image.
        prob (list): A list of the probabilities corresponding to the predicted categories.
    """

    model = load_checkpoint(checkpoint)
    prob, classes = predict(image_path, model)
    labels = model.class_to_idx
    label_mapping = list()
    for i in classes:
        label_mapping.append(list(labels.keys())[i])

    category_mapping = [load_mapping('cat_to_name.json')[str(cls)] for cls in label_mapping]
    return category_mapping, prob


def show_top5(image_path,checkpoint):
    """
    Show the top 5 predictions for the given image.

    Parameters:
    - image_path (str): The path to the image file.
    - chns:
    Noneeckpoint (str): The path to the checkpoint file.

    Retur
    """
        
    model = load_checkpoint(checkpoint)
    prob, classes = predict(image_path, model)
    labels = model.class_to_idx
    label_mapping = list()
    for i in classes:
        label_mapping.append(list(labels.keys())[i])

    category_mapping = [load_mapping('cat_to_name.json')[str(cls)] for cls in label_mapping]
    im = Image.open(image_path)
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(im)
    y_position = np.arange(len(category_mapping))
    ax[1].barh(y_position, prob)
    ax[1].set_yticks(y_position)
    ax[1].set_yticklabels(category_mapping)
    ax[1].invert_yaxis()
    ax[1].set_xlabel('Accuracy (%)')
    ax[0].set_title('Top 5 Predictions')



def get_args_2():
    """
    Parse command line arguments for the function and return the parsed arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagepath', type=str)
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth')
    

    in_args = parser.parse_args()
    return in_args


def check_command_line_args_predict(in_args):
    """
    Check the command line arguments for prediction.

    Args:
        in_args: The command line arguments.

    Returns:
        None.
    """

    
    if in_args is None:
        print ('No command line arguments')
    else:
        print('Command Line Arguments:\n    checkpoint = ', in_args.checkpoint, '\n    imagepath = ',
               in_args.imagepath)

def main():
  """
  The `main` function is the entry point of the program. It does the following:
  
  1. Retrieves command line arguments using the `get_args_2` function.
  2. Calls the `check_command_line_args_predict` function to validate the command line arguments.
  3. Retrieves the `checkpoint` and `image_path` from the command line arguments.
  4. Calls the `print_top5` function with the `image_path` and `checkpoint` as arguments, and assigns the returned values to `top_5` and `prob`.
  5. Formats the elements of `prob` as percentages and assigns the result to `prob2`.
  6. Prints the `top_5` and their corresponding probabilities.
  7. Prints the name of the first element in `top_5` and its corresponding probability.
  """
  
  in_args = get_args_2()

  check_command_line_args_predict(in_args)
  checkpoint = in_args.checkpoint
  image_path = in_args.imagepath
  top_5, prob =  print_top5(image_path= image_path ,checkpoint= checkpoint)

  prob2 = ['{:.2%}'.format(float) for float in prob]
  print(top_5, 'with probability: ', prob2 )
  print('Flower name is: ',  top_5[0],  ' with probability: ' ,prob2[0] )

# Call to main function to run the program
if __name__ == "__main__":
    main()
