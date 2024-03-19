import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cycler

def visualize_embeddings(embeddings,
                         labels,
                         label_mappings,
                         save_as=None,
                         show=False,
                         ax = None,
                         markersize=5):
    label_set = np.unique(labels)
    num_classes = len(label_set)
    
    if ax is None:
      plt.figure(figsize=(10,10))
      ax = plt.gca() 
    ax.set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0,0.9, num_classes)]
        )
    )

    for i in range(num_classes):
      idx = (labels==label_set[i])
      ax.plot(embeddings[idx,0], embeddings[idx,1], ".", markersize=markersize, label=label_mappings[label_set[i]])
    ax.legend(loc="best", markerscale=1)
    if save_as is not None:
      plt.savefig(save_as)
    if show == True: 
      plt.show()

def visualize_embeddings3D(embeddings,
                           labels,
                           label_mappings,
                           save_as=None,
                           show=False,
                           markersize=5):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(projection='3d')
    label_set = np.unique(labels)
    num_classes = len(label_set)
    ax.set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0,0.9, num_classes)]
        )
    )

    for i in range(num_classes):
        idx = (labels==label_set[i])
        print(len(idx))
        plt.plot(embeddings[idx,0],embeddings[idx,1],embeddings[idx,2], ".", markersize=markersize, label=label_mappings[label_set[i]])
    plt.legend(loc="best", markerscale=1)
    if save_as is not None:
      plt.savefig(save_as)
    if show == True: 
      plt.show()

def compare_embeddings(embeddings1,
                       embeddings2,
                       labels1,
                       labels2,
                       embeddings3=None,
                       labels3=np.zeros(0),
                       label_mappings=None,
                       save_as=None,
                       show=False,
                       markersize=5):
    if embeddings3 is not None and labels3 is not None:
      fig,(ax1,ax2,ax3) = plt.subplots(1,3)
      fig.set_size_inches(21,7)
    else:
      fig,(ax1,ax2) = plt.subplots(1,2)
      fig.set_size_inches(20,15)

    if label_mappings is None:
      label_mappings = {l:l for l in np.concatenate((labels1,labels2,labels3))}
    label_set = list(label_mappings.keys())
    num_classes = len(label_set)
      
    ax1.set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0,0.9, num_classes)]
        )
    )
    ax2.set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0,0.9, num_classes)]
        )
    )
    if embeddings3 is not None and labels3 is not None: 
      ax3.set_prop_cycle(
          cycler(
              "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0,0.9, num_classes)]
          )
      )

    for i in range(num_classes):
      idx = (labels1==label_set[i])
      ax1.plot(embeddings1[idx,0], embeddings1[idx,1], ".", markersize=markersize, label=label_mappings[label_set[i]])

      idx = (labels2==label_set[i])
      ax2.plot(embeddings2[idx,0], embeddings2[idx,1], ".", markersize=markersize, label=label_mappings[label_set[i]])
      if embeddings3 is not None and labels3 is not None: 
        idx = (labels3==label_set[i])
        ax3.plot(embeddings3[idx,0], embeddings3[idx,1], ".", markersize=markersize, label=label_mappings[label_set[i]])
    ax2.legend(loc="best", markerscale=1)
    if save_as is not None:
      plt.savefig(save_as)
    if show == True: 
      plt.show()

def show_kernels(clf,embeddings,labels,label_mappings=None,save_as=None,show=False):
    if label_mappings is None:
      label_mappings = {l:l for l in np.concatenate(label_mappings)}
    train_probs = clf.predict_proba(embeddings)
    plt.figure()
    sns.set_style('whitegrid')
    h_mu,h_std = train_probs[labels==0][:,0].mean(),train_probs[labels==0][:,0].std()
    f_mu,f_std = 1-train_probs[labels==1][:,1].mean(),train_probs[labels==1][:,1].std()

    sns.kdeplot(np.random.normal(h_mu,h_std,1000))
    sns.kdeplot(np.random.normal(f_mu,f_std,1000))
    plt.legend([label_mappings[0],label_mappings[1]])
    if save_as is not None:
      plt.savefig(save_as)
    if show == True: 
      plt.show()
