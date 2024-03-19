import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import wandb
from tensorboardX import SummaryWriter
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from ..utils import EarlyStopper

# EEG/EMG prototypical Model
class PrototypicalModel(nn.Module):
    
    def __init__(self,
                 net,
                 metric: Union[str,nn.Module] = None,
                 distance: nn.Module = None,
                 mean: nn.Module = None) -> None:
        super().__init__()

        self.net = net
        
        if metric == 'euclidean':
            self.distance = self.__euclidean_distance__
            self.mean = None #TODO future work will imply new metrics -> new possible way of computing means
        elif metric =='cosine':
            self.distance = self.__cosine_distance__
            self.mean = None
        elif distance is not None and mean is not None:
            self.distance = distance
            self.mean = mean
        else:
            raise ValueError('Please specify a metric between \'euclidean\' or \'cosine\', or provide custom modules for distance and mean')

    def compute_prototypes(self, support: Tensor, label: Tensor) -> Tensor:
        """Set the current prototypes used for classification.

        Parameters
        ----------
        data : torch.Tensor
            Input encodings
        label : torch.Tensor
            Corresponding labels

        """
        means_dict: Dict[int, Any] = {}
        for i in range(support.size(0)):
            means_dict.setdefault(int(label[i]), []).append(support[i])

        means = []
        n_means = len(means_dict)

        for i in range(n_means):
            # Ensure that all contiguous indices are in the means dict
            supports = torch.stack(means_dict[i], dim=0)
            if supports.size(0) > 1:
                mean = supports.mean(0).squeeze(0)
            else:
                mean = supports.squeeze(0)
            means.append(mean)

        prototypes = torch.stack(means, dim=0)
        return prototypes
    
    def compute_embeddings(self,
                           query : Tensor,
                           ):
      self.eval()
      with torch.no_grad():
        return self.net(query)
      
    def forward(self,  # type: ignore
                query_input: Tensor,
                support_input: Optional[Tensor] = None,
                support_label: Optional[Tensor] = None,
                prototypes: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Run a forward pass through the network.
        
        Parameters
        ----------
        query: Tensor
            The query examples, as tensor of shape (seq_len x batch_size)
        support: Tensor
            The support examples, as tensor of shape (seq_len x batch_size)
        support_label: Tensor
            The support labels, as tensor of shape (batch_size)

        Returns
        -------
        Tensor
            If query labels are

        """

        query_encoding = self.net(query_input)
        
        if prototypes is not None:
            prototypes = prototypes
        elif support_input is not None and support_label is not None:
            support_encoding = self.net(support_input)
            prototypes = self.compute_prototypes(support_encoding, support_label)
        else:
          raise ValueError("No prototypes set or support vectors have been provided")

        dist = self.distance(query_encoding, prototypes)

        return - dist
    
    def __euclidean_distance__(self, mat_1: Tensor, mat_2: Tensor):
        _dist = [torch.sum((mat_1 - mat_2[i])**2, dim=1) for i in range(mat_2.size(0))]
        dist = torch.stack(_dist, dim=1)
        return dist
    
    def __cosine_distance__(self, mat_1: Tensor, mat_2: Tensor):
        _dist = [F.cosine_similarity(mat_1,mat_2[i].unsqueeze(0),1) for i in range(mat_2.size(0))]
        dist = torch.stack(_dist, dim=1)
        return 1-dist
    
def get_all_embeddings(sampler, model,device):
  embeddings = []
  labels = []
  model.eval()
  with torch.no_grad():
    for _,(input,label), in enumerate(tqdm(sampler)):
      labels.append(label)
      input = input.to(device)
      embeddings.append(model.compute_embeddings(input))
  return torch.vstack(embeddings), torch.cat(labels)

def train_step(model,train_sampler,optimizer,loss_fn, writer, scheduler = None, log_interval = 0,max_grad_norm=0,device='cuda'):        	
    global global_step

    model.train()
    with torch.enable_grad():
        for idx,batch in enumerate(tqdm(train_sampler,position=1,leave=False)):
            # Zero the gradients and clear the accumulated loss
            optimizer.zero_grad()

            # Move to device
            batch = tuple(t.to(device) for t in batch)
            query_input,query_label,support_input,support_label = batch

            # Compute loss
            pred = model(query_input,support_input,support_label)
            loss = loss_fn(pred, query_label)
            loss.backward()

            # Clip gradients if necessary
            if max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimize
            optimizer.step()
            # Log training loss
            train_loss = loss.item()
            if log_interval > 0 and global_step % log_interval == 0:
                writer.add_scalar('Training/Loss_IT', train_loss, global_step)
            
            # Increment the global step
            global_step+=1

            # Zero the gradients when exiting a train step
            optimizer.zero_grad()

    return loss.item()


def test_step(model,train_eval_sampler,val_sampler,loss_fn,device): 
  model.eval()
  with torch.no_grad():

      # First compute prototypes over the training data
      embeddings, labels = [], []
      for batch in tqdm(train_eval_sampler,leave=False):
          source_input, target = tuple(t.to(device) for t in batch)
          embedding = model.compute_embeddings(source_input)
          labels.append(target.cpu())
          embeddings.append(embedding.cpu())
      # Compute prototypes
      embeddings = torch.cat(embeddings, dim=0)
      labels = torch.cat(labels, dim=0)
      prototypes = model.compute_prototypes(embeddings, labels).to(device)

      _preds, _targets = [], []
      for batch in tqdm(val_sampler,leave=False):
          # Move to device
          source_input, target = tuple(t.to(device) for t in batch)

          pred = model(source_input, prototypes=prototypes)
          _preds.append(pred.cpu())
          _targets.append(target.cpu())

      preds = torch.cat(_preds, dim=0)
      targets = torch.cat(_targets, dim=0)
      val_loss = loss_fn(preds, targets).item()
      
      val_metric = (pred.argmax(dim=1) == target).float().mean().item()
  return val_loss,val_metric

# Training Loop
def train(model,
          episodic_sampler,
          train_sampler,
          val_sampler,
          num_epochs,
          optimizer = None,
          scheduler = None,
          device='cuda',
          log_dir='.',
          log_interval=1,
          max_grad_norm=1,
          output_dir='model',
          es_patience = 0,
          es_min_delta = 0,
          use_wandb = False,
          )-> None:
    early_stopper = EarlyStopper(es_patience, min_delta=es_min_delta)
    if use_wandb:
        wandb.init( 
            project='EEG metric learning',
            name='test',
            entity="tcortecchia",
            config = {"n_epochs":num_epochs, 
                      "n_support":episodic_sampler.n_support, 
                      "n_episodes": episodic_sampler.n_episodes,
                      "n_classes": episodic_sampler.n_classes,
                      "eval_batch_size": val_sampler.batch_size,
                      "optimizer": optimizer,
                      "scheduler":scheduler})
    global global_step

    global_step = 0
    checkdir = os.path.join(output_dir,'checkpoints')
    # Training procedure set up
    best_metric = None
    best_model: Dict[str, torch.Tensor] = dict()
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(checkdir):
        os.makedirs(checkdir)
    writer = SummaryWriter(log_dir=log_dir
            )
    loss_fn = torch.nn.CrossEntropyLoss()
    
    if optimizer is None:
        print("No optimizer provided. Defaulting to Adam with lr=0.01.")
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.01)
    if scheduler is None:
        print("No lr scheduler provided.")

    # Training start
    print('Beginning training')
    for epoch in tqdm(range(num_epochs),position=0):
        train_loss = train_step(model,episodic_sampler,optimizer,loss_fn,writer,scheduler,log_interval,max_grad_norm,device)
        val_loss,val_metric = test_step(model,train_sampler,val_sampler,loss_fn,device)
        if scheduler: 
            lr = scheduler.get_last_lr()
            scheduler.step()
        else:
            lr = optimizer.param_groups[0]['lr']
        
        # Update best model
        if best_metric is None or val_metric > best_metric:
            best_metric = val_metric
            best_model_state = model.state_dict()
            for k, t in best_model_state.items():
                best_model_state[k] = t.cpu().detach()
            best_model = best_model_state
            torch.save({'model':best_model,
                        'epoch':epoch,
                        'optimizer':optimizer.state_dict()}, os.path.join(checkdir, f'epoch{epoch+1}_{best_metric:0.4f}.pt'))

        # Log metrics
        tqdm.write(f'Epoch {epoch+1}/{num_epochs}: Train loss: {train_loss} - Val loss: {val_loss} - Val acc: {val_metric}')
        writer.add_scalar('Hyperparameters/Learning_Rate', lr, epoch)
        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', val_metric, epoch)

        if use_wandb: wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_metric, "lr": lr})
        if early_stopper.early_stop(validation_loss=val_loss): 
            tqdm.write('Early stopping')
            break
    # Save the best model
    print("Finished training.")

    torch.save(best_model, os.path.join(output_dir, 'model.pt'))

    if use_wandb:
          wandb.finish()