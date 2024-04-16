import glob
import json
import logging
import os
import shutil
from datetime import datetime
from timeit import default_timer

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

logger = logging.getLogger("Trainer")

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    dataloader: tuple,
    device: str = 'cuda',
):
    model.train()

    total_num_samples = 0
    avg_acc = 0
    avg_loss = 0

    start = default_timer()

    for x,y in tqdm(dataloader,position=1, leave=False):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        # Forward pass
        logits = model(x).squeeze()

        # Compute loss
        loss = criterion(logits, y)

        # Compute prediction
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        num_samples = y.numel()
        avg_loss += loss.detach().cpu().item() * num_samples
        avg_acc += acc.cpu().item() * num_samples
        total_num_samples += num_samples

        optimizer.zero_grad()


    stop = default_timer()

    avg_loss = avg_loss / total_num_samples
    avg_acc = avg_acc / total_num_samples
    epoch_time = stop - start


    return avg_loss, avg_acc, epoch_time

def validate_one_step(model: nn.Module,
                      criterion: nn.Module,
                      dataloader: tuple,
                      device: str = 'cuda',):
    
    if dataloader is None:
        return np.nan, np.nan
    
    model.eval()

    avg_loss = 0
    avg_acc = 0

    total_num_samples = 0
    with torch.no_grad():
        for x,y in tqdm(dataloader,position=1, leave=False):
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            logits = model(x).squeeze(1)
            # Compute loss
            loss = criterion(logits, y)

            # Compute prediction
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean()

            # Update metrics
            num_samples = y.numel()
            avg_loss += loss.detach().cpu().item() * num_samples
            avg_acc += acc.cpu().item() * num_samples
            total_num_samples += num_samples

    avg_loss = avg_loss / total_num_samples
    avg_acc = avg_acc / total_num_samples


    return avg_loss, avg_acc

class StreamingJSONWriter:
    """
    Serialize streaming data to JSON.

    This class holds onto an open file reference to which it carefully
    appends new JSON data. Individual entries are input in a list, and
    after every entry the list is closed so that it remains valid JSON.
    When a new item is added, the file cursor is moved backwards to overwrite
    the list closing bracket.
    """

    def __init__(self, filename, encoder=json.JSONEncoder):
        if os.path.exists(filename):
            self.file = open(filename, "r+")
            self.delimeter = ","
        else:
            self.file = open(filename, "w")
            self.delimeter = "["
        self.encoder = encoder

    def dump(self, obj):
        """
        Dump a JSON-serializable object to file.
        """
        data = json.dumps(obj, cls=self.encoder)
        close_str = "\n]\n"
        self.file.seek(max(self.file.seek(0, os.SEEK_END) - len(close_str), 0))
        self.file.write(f"{self.delimeter}\n    {data}{close_str}")
        self.file.flush()
        self.delimeter = ","

    def close(self):
        self.file.close()

class MyTrainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 es_patience: int,
                 es_min_delta: float,
                 max_num_ckpts: int = -1) -> None:
        
        """
        Parameters:
        model: gcn model
        optimizer: torch Optimizer
        scheduler: learning rate scheduler
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.early_stopper = None
        if es_patience > 0 :
             self.early_stopper = EarlyStopper(es_patience,es_min_delta)

        self.max_num_ckpts = max_num_ckpts

    def train(
        self,
        num_epochs: int,
        train_loss_func: nn.Module,
        train_loader: tuple,
        val_loader: tuple = None,
        logger_name: str = None,
        device: str = 'cpu',
        path: str = None,
    ) -> None:

        if path is None: path = os.getcwd()
        if not os.path.exists(path):
             os.mkdir(path)

        # Set a path for best model and checkpoints,create them if missing
        outpath = os.path.join(path, "BestModel")
        checkpath = os.path.join(path, "CheckPoints")
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        os.mkdir(outpath)    
        if os.path.exists(checkpath):
            shutil.rmtree(checkpath)
        os.mkdir(checkpath)
        if logger_name is None:
             logger_name=datetime.now().strftime('%Y%m%d%H%M%S.log')

        # Set up logger
        jsonlog = StreamingJSONWriter(filename=logger_name)
        logger.info("## Training started ##")

        # Start training
        best_loss = np.inf
        for epoch in tqdm(range(num_epochs),position=0):

            train_loss,train_acc, train_time = train_one_step(
                self.model,
                self.optimizer,
                train_loss_func,
                train_loader,
                device = device,
                )      
           
            val_loss,val_acc = validate_one_step(self.model,
                train_loss_func,
                val_loader,
                device = device,)

            self.scheduler.step()
            logger.info(
                f"Epoch: {epoch + 1:03} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}"
            )

            if val_loss < best_loss:
                if self.max_num_ckpts !=0:
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                        },
                        checkpath + "/%05d" % (epoch + 1) + "-%6.5f" % (val_loss) + ".pt",
                    )
                if self.max_num_ckpts > 0:
                    ckpts = glob.glob(f'{checkpath}/*.pt')
                    if len (ckpts) > self.max_num_ckpts:
                        os.remove(sorted(ckpts,key=lambda x: os.stat(x).st_ctime)[0])

                log_dict = {
                    "Epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc":train_acc,
                    "val_loss":val_loss,
                    "val_acc":val_acc,
                    "train_time": train_time,
                }

                jsonlog.dump(log_dict)
                best_loss = train_loss
                torch.save({"model": self.model.state_dict()}, outpath + "/best-model.pt")

            monitor_loss = val_loss if not np.isnan(val_loss) else train_loss
            
            if self.early_stopper is not None and self.early_stopper.early_stop(validation_loss=monitor_loss):
                tqdm.write('Early Stopping')
                break
            
            tqdm.write(f'Epoch {epoch+1}/{num_epochs}\t Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}')
        jsonlog.close()
        logger.info("## Training complete ##")

