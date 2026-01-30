from importlib import import_module

import numpy as np
import yaml
from prettytable import PrettyTable
from torch import load
from torch.utils.data import Dataset


def fix_mat(data):
    """ Recursively fix MATLAB structs loaded with scipy.io.loadmat by removing unnecessary nesting and squeezing arrays."""
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
    """ Print a summary of the model's parameters in a tabular format."""
    
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

def make_montage(subset_channels=None):

    """ Create a dig montage with a (optional) subset of 10-20 standard EEG channels.
    Args:
        subset_channels (list,optional): List of channel names to include in the montage.
    Returns:
        montage (mne.channels.DigMontage): Montage object with the specified channels.
    """

    from mne.channels import make_dig_montage, make_standard_montage

    # Load the full 10-20 montage
    full_montage = make_standard_montage('standard_1020')

    if subset_channels is None:
        return full_montage

    # Extract only the positions of the subset channels
    all_positions = full_montage.get_positions()['ch_pos']
    subset_positions = {ch.upper(): pos for ch, pos in all_positions.items() if ch.upper() in subset_channels}

    # Order the subset positions based on the order in subset_channels
    subset_positions = {ch: subset_positions[ch] for ch in subset_channels}

    # Create a new montage using only those positions
    return make_dig_montage(ch_pos=subset_positions, coord_frame='head')
    
    
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
    

def fidx(channels, chlbl):
    """ Find the indices of specified channels in a list of channel labels.
    Args:
        channels (list): List of channel names to find.
        chlbl (list): List of available channel labels.
    Returns:
        idx (np.array): Array of indices corresponding to the specified channels.
    """

    chlbl = np.array(chlbl)
    idx = np.zeros(len(channels),dtype = int)
    for i,ch in enumerate(channels):
        if ch in chlbl:
            idx[i] = np.where(chlbl == ch)[0].item()
        else:
            idx[i] = -1
            raise Exception(f"Channel {ch} not found in channels")
    return idx