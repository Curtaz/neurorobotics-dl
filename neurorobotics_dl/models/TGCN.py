import torch
from torch import nn
import torch.nn.functional as F

class GraphConvWithLearnableWeight(nn.Module):
    def __init__(self, num_channels, in_features, out_features):
        super(GraphConvWithLearnableWeight, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        ## Not learnable support buffers
        Aui = torch.triu(torch.ones(num_channels,num_channels),1) # Upper triangular
        self.register_buffer('Aui',Aui)
        Al = torch.tril(torch.ones(num_channels,num_channels),-1) # Lower triangular
        self.register_buffer('Al',Al)
        EyeA = torch.eye(num_channels) # Fixed diagonal
        self.register_buffer('EyeA',EyeA)

        ## Learnable adjacency weights
        self.Wa = nn.Parameter(self.weight_init(num_channels),requires_grad=True)

        ## Graph Convolution
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def weight_init(self,num_channels):
        a =  torch.ones(num_channels,num_channels)
        row,col = torch.triu(torch.ones(num_channels,num_channels),1).nonzero().T
        Wa = a[row,col]
        return Wa.flatten()
    
    def forward(self, x):
        adj = self.Aui.clone()
        adj[adj>0] = self.Wa
        adj = adj + self.Al*adj.T + self.EyeA

        # # Compute degree matrix
        # degree_matrix = torch.sum(adj, dim=1)
        # # degree_matrix = torch.sum(adj, dim=1)
        # degree_matrix = torch.diag(degree_matrix.pow(-0.5))
        
        # # Normalize adjacency matrix
        # adj= torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)

        # Perform graph convolution
        support = torch.matmul(adj, x)

        # Perform graph convolution
        support = torch.matmul(x, self.weight) + self.bias
        output = torch.matmul(adj, support)
        return torch.relu(output)
    
    def get_adjacency(self):
        with torch.no_grad():
            W = self.Aui.clone()
            W[W>0] = self.Wa
            W = W + self.Al*W.T + self.EyeA
            # # Compute degree matrix
            # degree_matrix = torch.sum(W, dim=1).abs()
            # # degree_matrix = torch.sum(adj, dim=1)
            # degree_matrix = torch.diag(degree_matrix.pow(-0.5))
        
            # # Normalize adjacency matrix
            # W= torch.matmul(torch.matmul(degree_matrix, W), degree_matrix)
            
            return W 

# class GraphConvWithLearnableWeight(nn.Module):
#     def __init__(self, num_channels, in_features, out_features):
#         super(GraphConvWithLearnableWeight, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
        
#         ## Not learnable support buffers
#         Aui = torch.triu(torch.ones(num_channels,num_channels),1) # Upper triangular
#         self.register_buffer('Aui',Aui)
#         Al = torch.tril(torch.ones(num_channels,num_channels),-1) # Lower triangular
#         self.register_buffer('Al',Al)
#         EyeA = torch.eye(num_channels) # Fixed diagonal
#         self.register_buffer('EyeA',EyeA)

#         ## Learnable adjacency weights
#         self.Wa = nn.Parameter(self.weight_init(num_channels),requires_grad=True)

#         ## Graph Convolution
#         self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
#         self.bias = nn.Parameter(torch.FloatTensor(out_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         nn.init.zeros_(self.bias)

#     def weight_init(self,num_channels):
#         return torch.ones(num_channels,num_channels) + torch.randn(num_channels,num_channels)
        
#     def forward(self, x):
#         W = self.Wa +self.Wa.T
        
#         with torch.no_grad():
#             adj = W
#             adj[adj!=0] = 1
#         # Compute degree matrix
#         degree_matrix = torch.sum(adj, dim=1).abs()
#         # degree_matrix = torch.sum(adj, dim=1)
#         degree_matrix = torch.diag(degree_matrix.pow(-0.5))

#         # Normalize adjacency matrix
#         adj= torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)
            
#         # Perform graph convolution
#         support = torch.matmul(x, self.weight) + self.bias
#         output = adj @ W @ support

#         return torch.relu(output)
    
#     def get_adjacency(self):
#         with torch.no_grad():
#             W = self.Wa

#             return W 
        
class GCN_GRU_sequence_fxdD(nn.Module):
    def __init__(self,
                 num_channels,
                 gcn_input_dim,
                 num_classes,
                 gcn_output_dim = 16,
                 gru_hidden_units = 32,
                 gcn_dropout = 0.3,
                 gru_dropout = 0.5,):

        super().__init__()
        """
        Input:
            x: (batch_size, num_channels, seq_len, num_features)
        """
        self.num_channels = num_channels

        self.conv = nn.Conv2d(1,gcn_input_dim,kernel_size=(1, 512),padding=(0, 512 // 2), bias = False)
        self.bn1 = nn.BatchNorm2d(gcn_input_dim)
        
        ## GCN here
        self.gcn = GraphConvWithLearnableWeight(num_channels,gcn_input_dim,gcn_output_dim)
        
        self.bn2 = nn.BatchNorm2d(num_channels)

        self.gcn_drop = nn.Dropout(gcn_dropout)

        ## EEG GRU
        self.gru = nn.GRU(num_channels*gcn_output_dim,gru_hidden_units,batch_first=True)
        # self.gru = nn.GRU(gcn_output_dim,gru_hidden_units,batch_first=True)

        self.gru_drop = nn.Dropout(gru_dropout)
        
        self.fc = nn.Linear(gru_hidden_units,num_classes)

    def forward(self,x):

        shape = x.shape
        if len(shape)<4: 
            x = x.view((1,*shape))
        b_s,n_ch,s_s,s_l = shape 

        h = self.conv(x)
        h = self.bn1(h)
        h = h.permute(0,3,2,1)

        # h = self.gcn(x)
        h = self.gcn(h)
        h = self.gcn_drop(h)
        # print(h.shape)
        h = h.permute(0,2,1,3)
        h = self.bn2(h)
        h = h.permute(0,2,1,3).flatten(2)
        # h = h.permute(0,2,1,3).mean(dim=2)
        _,h = self.gru(h)
        out = self.gru_drop(h)
        out = out.view(b_s,-1)
        out = self.fc(out)
        # out = F.softmax(out,dim=-1)
        return out
    
class GCN_dGRU_sequence_fxdD(nn.Module):
    def __init__(self,
                num_channels_eeg,
                num_channels_emg,
                gcn_input_dim,
                num_classes,
                gcn_output_dim = 16,
                gru_hidden_units = 32,
                gcn_dropout = 0.3,
                gru_dropout = 0.5,
                ):
        
        super().__init__()

        self.num_EEG = num_channels_eeg
        self.num_EMG = num_channels_emg
        self.num_tot = num_channels_eeg + num_channels_emg

        # self.conv = nn.Conv2d(1,gcn_input_dim,kernel_size=(1, conv_kernLength),padding=(0, conv_kernLength // 2), bias = False)

        ## GCN here
        self.gcn = GraphConvWithLearnableWeight(self.num_tot,gcn_input_dim,gcn_output_dim)

        self.bn1 = nn.BatchNorm2d(self.num_tot)

        self.gcn_drop = nn.Dropout(gcn_dropout)

        ## EEG GRU
        self.gru_EEG = nn.GRU(num_channels_eeg*gcn_output_dim,gru_hidden_units,batch_first=True)
        
        ## EMG GRU
        self.gru_EMG = nn.GRU(num_channels_emg*gcn_output_dim,gru_hidden_units,batch_first=True)

        self.gru_drop = nn.Dropout(gru_dropout)

        self.fc = nn.Linear(2*gru_hidden_units,num_classes)
        
    def forward(self,x):

        shape = x.shape
        if len(shape)<4: 
            x = x.view((1,*shape))
        b_s,n_ch,s_s,s_l = shape 

        h = self.gcn(x)
        h = self.gcn_drop(h)
        # print(h.shape)
        h = h.permute(0,2,1,3)
        h = self.bn1(h)
        h = h.permute(0,2,1,3)
        
        h_eeg = h[:,:,:self.num_EEG,:].flatten(2)
        h_emg = h[:,:,self.num_EEG:,:].flatten(2)
        _,h_eeg = self.gru_EEG(h_eeg)

        _,h_emg = self.gru_EMG(h_emg)


        out = self.gru_drop(torch.cat((h_eeg,h_emg),axis=-1))
        out = out.view(b_s,-1)
        

        out = F.softmax(self.fc(out),dim=-1)
        return out

    def get_adjacency(self):
        return self.gcn.get_adjacency()
