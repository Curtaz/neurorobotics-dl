from importlib import import_module

import numpy as np
import yaml
from prettytable import PrettyTable
from torch.utils.data import Dataset
from torch import load

def fix_mat(data):
    if data.dtype.names:
        new_data = dict()
        for name in data.dtype.names:
            new_data[name]=data[0][name][0]
        for k,v in new_data.items():
            if v.dtype.names:
                new_data[k] = fix_mat(v)
            else:
                new_data[k] = np.squeeze(v)
        return new_data
    else:
        return data

def summary(model):
    print('Model Parameters:')
    model_table = PrettyTable()
    model_table.field_names = ["Layer", "Params", "Shape", "Trainable"]
    tot_params = 0
    for name,param in model.named_parameters():
        model_table.add_row([name,param.numel(),param.shape,param.requires_grad])
        tot_params += param.numel()
    model_table.add_row(['Total',tot_params,'',''])
    print(model_table)

def write_to_yaml(filename,data: dict):
    """
    Write the given dictionary to a YAML file.

    Args:
        filename (str): The name of the file to write to.
        data (dict): The dictionary to be written to the file.

    Returns:
        None
    """
    with open(filename, "a") as outfile:
        yaml.dump(data, outfile)

def get_class(class_str):
    """
    Import and return a class based on its string representation.

    Args:
        class_str (str): The string representation of the class to be imported.
            It should be in the format "module_path.class_name".

    Returns:
        The imported class.

    Raises:
        ImportError: If the class cannot be imported or does not exist.
    """
    try:
        module_path, class_name = class_str.strip().rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(class_str)

def load_checkpoint(model_ckpt,*extras):
    """
    Load a pretrained model from a checkpoint file.

    Args:
        model_ckpt (str): The path to the model checkpoint file.
        *extras (str): Optional list of extra parameters to load from the checkpoint.

    Returns:
        model (nn.Module): The pretrained model.
        extras (dict): The dictionary of extra parameters, if `extras` is provided.

    This function loads a pretrained model from a checkpoint file and returns the model. If `extras` is provided,
    it also loads the specified extra parameters from the checkpoint and returns them in a dictionary.

    Example:
        model, extras = load_checkpoint('path/to/checkpoint.pt', 'param1', 'param2')
    """

    state_dict = load(model_ckpt)
    model = get_class(state_dict['config']['classname'])(**state_dict['config']['options'])
    model.load_state_dict(state_dict['model'])
    if len(extras)>0:
        extras = {k:state_dict[k] for k in extras}
        return model,extras
    return model

"""__________________________________________________UTILITIES________________________________________________"""

class MyDataset(Dataset):
    def __init__(self,X,y):
        """
        Initializes an instance of the MyDataset class for model training.

        Args:
            X (torch.Tensor or numpy.ndarray): The input data.
            y (torch.Tensor or numpy.ndarray): The target data.
        """
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index],self.y[index]
    
    def __len__(self):
        return len(self.X)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0,is_inverted = False):
        """
        Initializes an instance of the EarlyStopper class.

        Args:
            patience (int, optional): The number of epochs to wait for improvement in validation loss before stopping training. Defaults to 1.
            min_delta (float, optional): The minimum change in validation loss to count as an improvement. Defaults to 0.
            is_inverted (bool, optional): Whether to invert the early stopping condition, i.e., stop training when validation loss improves instead of decreasing. Defaults to False.

        Returns:
            None
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.is_inverted = is_inverted
        

    def early_stop(self, validation_loss):
        if self.is_inverted:
          if validation_loss > self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
          elif validation_loss < (self.min_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False