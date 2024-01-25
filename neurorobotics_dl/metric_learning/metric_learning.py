import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

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

def train_step(model,train_sampler,optimizer,loss_fn,writer,log_interval = 0,max_grad_norm=0,device='cuda'):        	
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

# train(model,episodic_sampler,val_sampler,device,log_dir='logs')

# Training Loop
def train(model,
          episodic_sampler,
          train_sampler,
          val_sampler,
          num_epochs,
          learning_rate=0.001,
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
            config = {	"lr": learning_rate,
                         "n_epochs":num_epochs, 
                        "n_support":episodic_sampler.n_support, 
                        "n_episodes": episodic_sampler.n_episodes,
                        "n_classes": episodic_sampler.n_classes,
                        "eval_batch_size": val_sampler.batch_size})
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
    parameters = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    # Training start
    print('Beginning training')
    for epoch in tqdm(range(num_epochs),position=0):
        train_loss = train_step(model,episodic_sampler,optimizer,loss_fn,writer,log_interval,max_grad_norm,device)
        val_loss,val_metric = test_step(model,train_sampler,val_sampler,loss_fn,device)
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
        writer.add_scalar('Hyperparameters/Learning Rate', lr, epoch)
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

    # # Model evaluation 
    # best_model = PrototypicalTransformerModel(model_ckpt,128).to(device)
    # best_model.load_state_dict(torch.load('/content/out/prototype/model.pt'))
    # best_model.eval()
    
    # # Visualize the new embeddings 
    # umap_visualizer = umap.UMAP()
    # embeddings, labels = get_all_embeddings(MyDataset(train_features), best_model,device)
    # embeddings_reduced = umap_visualizer.fit_transform(embeddings.cpu().numpy())
    # fig = visualize_embeddings(embeddings_reduced, labels.cpu().numpy(),artists_mappings)