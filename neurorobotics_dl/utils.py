from importlib import import_module

import numpy as np
import yaml
from prettytable import PrettyTable
from torch.utils.data import Dataset

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
    with open(filename, "a") as outfile:
        yaml.dump(data, outfile)

def get_class(class_str):
    try:
        module_path, class_name = class_str.strip().rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(class_str)

"""__________________________________________________UTILITIES________________________________________________"""

class MyDataset(Dataset):
    def __init__(self,X,y):
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index],self.y[index]
    
    def __len__(self):
        return len(self.X)