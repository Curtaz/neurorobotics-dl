import torch
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict, Any, Optional, Union

class EEGNet(nn.Module):
        def __init__(self,
                     nb_classes, 
                     Chans = 16, 
                     Samples = 1024,
                     dropoutRate = 0.5,
                     kernLength = 512,
                     F1 = 8, 
                     D = 2, 
                     F2 = 16, 
                     norm_rate = 0.25,
                     dropoutType = 'Dropout'
                      ) -> None:
                super().__init__()

                self.conv = nn.Conv2d(1,F1,kernel_size=(1, kernLength),padding=(0, kernLength // 2), bias = False)
                self.bn1 = nn.BatchNorm2d(F1)
                self.depth_conv = nn.Conv2d(F1,F1 * D, kernel_size=(Chans,1), groups=F1,bias=False)
                self.bn2 = nn.BatchNorm2d(F1*D)
                self.drop1 = nn.Dropout2d(p=dropoutRate)

                self.point_conv = nn.Conv2d(F1 * D, F2, kernel_size=(1,16),padding=(0, 8), bias=False)
                self.bn3 = nn.BatchNorm2d(F2)
                self.drop2 = nn.Dropout2d(p=dropoutRate)
                self.fc = nn.Linear(F2*Samples//32,nb_classes)

        def forward(self,x):
                h = self.conv(torch.permute(x,(0,2,1,3)))
                h = self.bn1(h)
                h = self.depth_conv(h)
                h = self.bn2(h)
                h = F.elu(h)
                h = F.avg_pool2d(h,(1, 4))
                h = self.drop1(h)
                h = self.point_conv(h)
                h = self.bn3(h)
                h = F.elu(h)
                h = F.avg_pool2d(h,(1, 8))
                h = self.drop2(h)
                h = torch.flatten(h,start_dim=1)
                h = self.fc(h)
                return h

class GCNWithLearnableWeight(nn.Module):
    def __init__(self,
                 num_channels,
                 input_dim,
                 hidden_dims,
                 output_dim,
                 activation=F.relu,
                 activate_last = True,
                 ) -> None:
        super().__init__()

        ## Not learnable support buffers
        Aui = torch.triu(torch.ones(num_channels,num_channels))-torch.eye(num_channels) # Upper triangular
        self.register_buffer('Aui',Aui)
        Al = torch.tril(torch.ones(num_channels,num_channels),-1) # Lower triangular
        self.register_buffer('Al',Al)
        EyeA = torch.eye(num_channels) # Fixed diagonal
        self.register_buffer('EyeA',EyeA)

        ## Learnable adjacency weights
        self.Wa = torch.randn((1,num_channels*(num_channels-1)//2)) + 1  # init weights as gaussian distribution with mean 1
        self.Wa.requires_grad=True
        self.Wa = nn.Parameter(self.Wa)

        ## Graph Convolutional Layers
        last_hidden_dim = input_dim
        self.convs = nn.Sequential()
        for dim in hidden_dims:
            self.convs.append(GCNConv(last_hidden_dim,dim))
            last_hidden_dim = dim
        self.output = GCNConv(last_hidden_dim,output_dim)
        self.activation = activation
        self.activate_last = activate_last
    def forward(self,x):

        ## Compute edge list and weights
        A = self.Aui.clone()
        A[A>0] = F.sigmoid(self.Wa)
        A = A + self.Al*A.T + self.EyeA

        edge_index = A.nonzero().T
        edge_weight = A.flatten()
        ## Execute hidden convolutions
        h = x
        for conv in self.convs:
            h = conv(h,
                     edge_index = edge_index,
                     edge_weight = edge_weight)
            h = self.activation(h)
            
        ## Execute last convolution
        o = self.output(h,
                        edge_index = edge_index,
                        edge_weight = edge_weight)
        if self.activate_last:
            o = self.activation(o)
        return o
    
    def get_adjacency(self):
        with torch.no_grad():
            ## Compute edge list and weights
            A = self.Aui.clone()
            A[A>0] = F.sigmoid(self.Wa)
            A = A + self.Al*A.T + self.EyeA
        return A.detach().cpu().numpy()

class GCN_GRU_sequence_fxdD(nn.Module):
    def __init__(self,
                 num_channels,
                 gcn_input_dim,
                 gcn_hidden_dims,
                 gcn_output_dim,
                 gcn_activation,
                 gru_hidden_units,
                 gcn_dropout,
                 gru_dropout):

        super().__init__()

        ## GCN here
        self.gcn = GCNWithLearnableWeight(num_channels,gcn_input_dim,gcn_hidden_dims,gcn_output_dim,gcn_activation,activate_last=True)
        
        self.batch_norm = nn.BatchNorm1d(num_channels)

        self.gcn_drop = nn.Dropout(gcn_dropout)

        ## EEG GRU
        self.gru = nn.GRU(num_channels*gcn_output_dim,gru_hidden_units,batch_first=True)
        self.gru_drop = nn.Dropout(gru_dropout)
        
    def forward(self,x):

        ## INPUT
        ## Add batch dimension if missing
        shape = x.shape
        if len(shape)<4: 
            x = x.view((1,*shape))
            b_s = 1
            n_ch,s_s,s_l = shape 
        else:
            b_s,n_ch,s_s,s_l = shape 

        # print('Input:',x.shape)
        x = x.permute(0,3,1,2).reshape(b_s*s_l,n_ch,s_s) #FIXME: check this is correct, find a better way
        # print('Before GCN:',x.shape)
        ## GCN
        x = self.gcn(x)

        x = self.batch_norm(x)
        x = self.gcn_drop(x)
        # print('After GCN:',x.shape)

        x = x.view(b_s,s_l,-1)
        # print('Before GRUs:',x.shape)
        
        _,h_eeg = self.gru(x)
        h_eeg = h_eeg.squeeze()
        # print('After GRU_EEG',h_eeg.shape)
        h_eeg = self.gru_drop(h_eeg)
        # return F.sigmoid(h_eeg)
        return h_eeg

class GCN_dGRU_sequence_fxdD(nn.Module):
    def __init__(self,
                 num_channels_eeg,
                 num_channels_emg,
                 conv_kernLength,
                 gcn_input_dim,
                 gcn_hidden_dims,
                 gcn_output_dim,
                 gcn_activation,
                 gcn_activate_last,
                 gru_hidden_units,
                 gcn_dropout,
                 gru_dropout):
        
        super().__init__()

        self.num_EEG = num_channels_eeg
        self.num_EMG = num_channels_emg
        self.num_tot = num_channels_eeg + num_channels_emg

        # self.conv = nn.Conv2d(1,gcn_input_dim,kernel_size=(1, conv_kernLength),padding=(0, conv_kernLength // 2), bias = False)
        # self.bn1 = nn.BatchNorm2d(gcn_input_dim)

        ## GCN here
        self.gcn = GCNWithLearnableWeight(self.num_tot,gcn_input_dim,gcn_hidden_dims,gcn_output_dim,gcn_activation,gcn_activate_last)
        
        self.batch_norm = nn.BatchNorm1d(self.num_tot)

        self.gcn_drop = nn.Dropout(gcn_dropout)

        ## EEG GRU
        self.gru_EEG = nn.GRU(num_channels_eeg*gcn_output_dim,gru_hidden_units,batch_first=True)
        self.gru_drop_EEG = nn.Dropout(gru_dropout)

        ## EMG GRU
        self.gru_EMG = nn.GRU(num_channels_emg*gcn_output_dim,gru_hidden_units,batch_first=True)
        self.gru_drop_EMG = nn.Dropout(gru_dropout)
        
    def forward(self,x):

        ## INPUT
        ## Add batch dimension if missing
        shape = x.shape
        if len(shape)<4: 
            x = x.view((1,*shape))
            b_s = 1
            n_ch,s_s,s_l = shape 
        else:
            b_s,n_ch,s_s,s_l = shape 

        print('Input:',x.shape)
        x = x.permute(0,3,1,2).reshape(b_s*s_l,n_ch,s_s) #FIXME: check this is correct, find a better way
        # print('Before GCN:',x.shape)
        ## GCN
        x = self.gcn(x)

        x = self.batch_norm(x)
        x = self.gcn_drop(x)
        # print('After GCN:',x.shape)

        x = x.view(b_s,s_l,n_ch,-1)
        # print('Before GRUs:',x.shape)

        ## EEG BRANCH
        h_eeg = x[:,:,:self.num_EEG,:].view(b_s,s_l,-1)
        # print("H_eeg:",h_eeg.shape)
        
        h_eeg = self.gru_EEG(h_eeg)
        h_eeg = self.gru_drop_EEG(h_eeg[1])
        # print('After GRU_EEG',h_eeg[1].shape)
        h_eeg = F.relu(h_eeg)
        
        ## EMG BRANCH
        h_emg = x[:,:,self.num_EEG:,:].view(b_s,s_l,-1)
        # print("H_emg:",h_emg.shape)

        h_emg = self.gru_EMG(h_emg)
        h_emg = self.gru_drop_EMG(h_emg[1])
        # print('After GRU_EMG',h_emg.shape)
        h_emg = F.relu(h_emg)

        ## CLASSIFICATION
        out = torch.cat((h_eeg,h_emg),axis=-1).view(b_s,-1)
        # print('After Concatenation',out.shape)

        return out

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

        query_encoding = self.compute_embeddings(query_input)
        
        if prototypes is not None:
            prototypes = prototypes
        elif support_input is not None and support_label is not None:
            support_encoding = self.compute_embeddings(support_input)
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