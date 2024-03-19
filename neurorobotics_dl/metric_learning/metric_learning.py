from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import torch
from neurorobotics_dl.utils import EarlyStopper
import wandb
import os
from typing import Dict
from tensorboardX import SummaryWriter
from torch.nn.functional import sigmoid

def get_all_embeddings(loader, model,device):
  embeddings = []
  labels = []
  model.eval()
  with torch.no_grad():
    for _,(input,label), in enumerate(tqdm(loader,leave=False)):
      labels.append(label.to(device))
      input = input.to(device)
      embeddings.append(model(input))
  return torch.vstack(embeddings), torch.cat(labels)


def test(model,train_loader, test_loader, distance,device):
  train_embeddings, train_labels = get_all_embeddings(train_loader, model,device)
  test_embeddings, test_labels = get_all_embeddings(test_loader, model,device)
  distance = distance.to(device)
  classes = train_labels.unique()
  centroids = torch.zeros(classes.shape[0],train_embeddings.shape[1],device = device)
  for cl in classes:
    centroids[cl,:] = train_embeddings[train_labels==cl].mean(axis=0)
  train_dist = distance(centroids,train_embeddings)
  test_dist = distance(centroids,test_embeddings)
#   print(train_dist)
#   print(distance.smallest_dist(train_dist,axis=0))

  accuracy_train = (distance.smallest_dist(train_dist,axis=0).indices==train_labels).float().mean()
  accuracy_test = (distance.smallest_dist(test_dist,axis=0).indices==test_labels).float().mean()
  return accuracy_train,accuracy_test

def train_step(model,
                train_loader,
                miner,
                optimizer,
                loss_fn,
                writer,
                log_interval = 0,
                max_grad_norm=0,
                device='cuda'):        	
    global global_step
    num_triplets = 0
    model.train()
    with torch.enable_grad():
        for idx, batch in enumerate(tqdm(train_loader,position=1,leave=False)):
            # Zero the gradients and clear the accumulated loss
            optimizer.zero_grad()
            # Move to device
            batch = tuple(t.to(device) for t in batch)
            (data, labels) = batch
            embeddings = sigmoid(model(data))
            hard_pairs = miner(embeddings, labels)

            loss = loss_fn(embeddings, labels, hard_pairs)
            loss.backward()

            # Clip gradients if necessary
            if max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimize
            optimizer.step()
            # Log training loss
            train_loss = loss.item()
            num_triplets+=miner.num_triplets
            if log_interval > 0 and global_step % log_interval == 0:
                writer.add_scalar('Training/Loss_IT', train_loss, global_step)
                writer.add_scalar('Training/Mined_triplets_IT',miner.num_triplets,global_step)
            
            # Increment the global step
            global_step+=1

            # Zero the gradients when exiting a train step
            optimizer.zero_grad()

    return loss.item(),num_triplets

def train(model,
          train_loader,
          val_loader,
          miner,
          loss_func,       
          distance,          
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
          use_wandb = False,):

    early_stopper = EarlyStopper(es_patience, min_delta=es_min_delta,is_inverted=True)
    if use_wandb:
        wandb.init( 
            project='EEG metric learning',
            name='test',
            entity="tcortecchia",
            config = {"n_epochs":num_epochs, 
                      'train_batch_size': train_loader.batch_size,
                      "eval_batch_size": val_loader.batch_size,
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
    
    if optimizer is None:
        print("No optimizer provided. Defaulting to Adam with lr=0.01.")
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.01)
    if scheduler is None:
        print("No lr scheduler provided.")


    # Training start
    print('Beginning training')
    for epoch in tqdm(range(num_epochs),position=0):

        train_loss,num_triplets = train_step(model,
                    train_loader,
                    miner,
                    optimizer,
                    loss_func,
                    writer = writer,
                    log_interval = log_interval,
                    max_grad_norm = max_grad_norm,
                    device=device)
        
        train_acc,val_acc = test(model,
                                train_loader,
                                val_loader,
                                distance,
                                device)

        if scheduler: 
            lr = scheduler.get_last_lr()
            scheduler.step()
        else:
            lr = optimizer.param_groups[0]['lr']
        
        # accuracy_calc = AccuracyCalculator()
        # Update best model
        if best_metric is None or val_acc > best_metric:
            best_metric = val_acc
            best_model_state = model.state_dict()
            for k, t in best_model_state.items():
                best_model_state[k] = t.cpu().detach()
            best_model = best_model_state
            torch.save({'model':best_model,
                        'epoch':epoch,
                        'optimizer':optimizer.state_dict()}, os.path.join(checkdir, f'epoch{epoch+1}_{best_metric:0.4f}.pt'))

        # Log metrics
        tqdm.write(f'Epoch {epoch+1}/{num_epochs}: Train loss: {train_loss} - Train acc: {train_acc} - Val acc: {val_acc} - Mined triplets: {num_triplets}')
        writer.add_scalar('Hyperparameters/Learning_Rate', lr, epoch)
        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Training/Accuracy', train_acc, epoch)
        writer.add_scalar('Validation/Accuracy', val_acc, epoch)
        writer.add_scalar('Training/Mined_triplets',num_triplets,epoch)


        if use_wandb: wandb.log({"train_loss": train_loss, "train_accuracy": train_acc, "val_accuracy": val_acc, "lr": lr})
        if early_stopper.early_stop(validation_loss=val_acc): 
            tqdm.write('Early stopping')
            break
    # Save the best model
    print("Finished training.")

    torch.save(best_model, os.path.join(output_dir, 'model.pt'))

    if use_wandb:
          wandb.finish()