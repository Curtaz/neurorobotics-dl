from prettytable import PrettyTable
import yaml
from pathlib import Path

def fix_mat(data):
    if data.dtype.names:
        new_data = dict()
        for name in data.dtype.names:
            new_data[name]=data[0][name][0]
        for k,v in new_data.items():
            if v.dtype.names:
                new_data[k] = fix_mat(v)
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
