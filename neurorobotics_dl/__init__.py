from prettytable import PrettyTable

def summary(model):
    print('Model Parameters: ')
    model_table = PrettyTable()
    model_table.field_names = ["Layer", "Params", "Shape", "Trainable"]
    tot_params = 0
    for name,param in model.named_parameters():
        model_table.add_row([name,param.numel(),param.shape,param.requires_grad])
        tot_params += param.numel()
    model_table.add_row(['Total',tot_params,'',''])
    print(model_table)