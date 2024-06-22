import torch
from torch import nn
from torch.nn import functional as F

class TC_Block(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 batch_norm=True, 
                 conv_kernel_size = 3,
                 pool_kernel_size = 2,
                 pool=None, 
                 activation=None,
                 dropout = 0,
                 dilation = 1):
        super(TC_Block,self).__init__()

        modules = []

        # Convolution
        modules.append(nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=(1,conv_kernel_size),
                                 padding=(0,conv_kernel_size//2*dilation),
                                 dilation=dilation))
        
        # Pooling
        if pool == 'avg':
            modules.append(nn.AvgPool2d((1,pool_kernel_size),stride=(1,pool_kernel_size)))
        elif pool == 'max':
            modules.append(nn.MaxPool2d((1,pool_kernel_size),stride=(1,pool_kernel_size)))
        elif pool is not None and pool != False:
            raise AttributeError(f'Invalid pooling {pool} mode specified')
        
        # Batch Norm
        if batch_norm:
            modules.append(nn.BatchNorm2d(out_channels))
        
        # Activation
        if activation is not None:
            modules.append(activation)
        
        # Dropout
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        # for l in self.conv:
        #     print(l._get_name(),end = ' ')
        #     x = l(x)
        #     print(x.shape)
        # return x
        return self.conv(x)

class CC_Block(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 batch_norm=True, 
                 conv_kernel_size = 3,
                 pool_kernel_size = 2,
                 pool=None, 
                 dropout = 0):
        super(CC_Block,self).__init__()

        modules = []
        modules.append(nn.Conv2d(in_channels,out_channels,kernel_size=(1,conv_kernel_size),padding=(0,conv_kernel_size//2),groups=120))
        if pool == 'avg':
            modules.append(nn.AvgPool2d((1,pool_kernel_size),stride=(1,pool_kernel_size)))
        elif pool == 'max':
            modules.append(nn.MaxPool2d((1,pool_kernel_size),stride=(1,pool_kernel_size)))
        elif pool is not None and pool != False:
            raise AttributeError(f'Invalid pooling {pool} mode specified')
        if batch_norm:
            modules.append(nn.BatchNorm2d(out_channels))
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        for l in self.conv:
            print(l._get_name(),end = ' ')
            x = l(x)
            print(x.shape)
        return x
        # return self.conv(x)
#%%
class AttentionHead(nn.Module):
    def __init__(self,
                 seq_len,
                 hidden_size) -> None:
        super().__init__()
        self.fck = nn.Linear(seq_len,hidden_size)
        self.fcq = nn.Linear(seq_len,hidden_size)
        self.fcv = nn.Linear(seq_len,seq_len)
    def forward(self,x):
        k = self.fck(x)
        q = self.fcq(x)
        v = self.fcv(x)
        phi = F.softmax(q.matmul(k.mT),dim=(1))

        x = torch.matmul(phi,v) +x
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,seq_len,hidden_size,num_heads) -> None:
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(seq_len,hidden_size) for _ in range(num_heads)])
    
    def forward(self,x):
        v = []
        for head in self.heads:
            v.append(head(x))
        v.append(x)
        return torch.cat(v,dim=1)


class TemporalAttention(nn.Module):
    def __init__(self,input_size):
        super(TemporalAttention,self).__init__()
        self.avg = nn.AvgPool2d(kernel_size=(1,5))
        self.max = nn.AvgPool2d(kernel_size=(1,5))
        self.conv1 = nn.Conv2d(2*input_size,input_size,(1,5),groups=input_size,padding = (0,2))
        self.conv2 = nn.Conv2d(input_size,2*input_size,(1,5),groups=input_size,padding = (0,2))
        self.splitsize = input_size
        self.adapt_avg = nn.AdaptiveAvgPool2d(1)
    
    def forward(self,x):
        h_avg = self.avg(x)
        h_max = self.max(x)
        i1 = torch.cat([h_avg,h_max],dim=1)
        i21 = F.elu(self.conv1(i1))
        i22 = self.conv2(i21)
        i31,i32= torch.split(i22,self.splitsize,1)
        i4 = i31+i32
        i5 = torch.sigmoid(self.adapt_avg(i4))
        return x*i5+x
    
class MSEM(nn.Module):
    def __init__(self,
                 input_size,
                 conv1_chans = 20,
                 conv2_chans =10,
                 L_kern = 25,
                 M_kern = 15,
                 S_kern=5) -> None:
        super().__init__()        
        self.tcl1 = TC_Block(input_size,conv1_chans,conv_kernel_size=L_kern,batch_norm=True,activation=nn.ELU())
        self.tcl2 = TC_Block(conv1_chans,conv2_chans,conv_kernel_size=L_kern,batch_norm=True,activation=nn.ELU(),dilation=3)
        self.tcm1 = TC_Block(input_size,conv1_chans,conv_kernel_size=M_kern,batch_norm=True,activation=nn.ELU())
        self.tcm2 = TC_Block(conv1_chans,conv2_chans,conv_kernel_size=M_kern,batch_norm=True,activation=nn.ELU(),dilation=3)
        self.tcs1 = TC_Block(input_size,conv1_chans,conv_kernel_size=S_kern,batch_norm=True,activation=nn.ELU())
        self.tcs2 = TC_Block(conv1_chans,conv2_chans,conv_kernel_size=S_kern,batch_norm=True,activation=nn.ELU(),dilation=3)

    def forward(self,x):
        f11 = self.tcl1(x)
        f12 = self.tcl2(f11)
        f21 = self.tcm1(x)
        f22 = self.tcm2(f21)
        f31 = self.tcs1(x)
        f32 = self.tcs2(f31)
        x = torch.cat([f11,f12,f21,f22,f31,f32,x],dim=1)
        return x

class C_Block(nn.Module):
    def __init__(self,
                 seq_len,
                 in_channels,
                 num_classes,
                 conv1_size = 40,
                 conv2_size = 20,
                 p1_kern = 5,
                 p2_kern = 2,
                 dropout = 0.5) -> None:
        super().__init__()
        self.conv1 = TC_Block(in_channels,conv1_size,batch_norm=True,pool='max',dropout=dropout,conv_kernel_size=5,pool_kernel_size=p1_kern,activation=nn.ELU())
        self.conv2 = TC_Block(conv1_size,conv2_size,batch_norm=True,pool='max',dropout=dropout,conv_kernel_size=5,pool_kernel_size=p2_kern,activation=nn.ELU())
        self.fc = nn.Linear(conv2_size*seq_len//(p1_kern*p2_kern),num_classes)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x,start_dim=1)
        x = torch.softmax(self.fc(x),dim=1)
        return x
    

class STANet(nn.Module):
    def __init__(self,
                 T,
                 C,
                 num_classes,
                 N=20,
                 L=1,
                 MSEM_C1=20,
                 MSEM_C2=10,
                 dropout=0.5) -> None:
        super().__init__()
        self.tc1 = TC_Block(1,N,pool='avg',conv_kernel_size=N,pool_kernel_size=2)
        self.mha = MultiHeadAttention(T//2,128,L)


        self.cc_block = nn.Sequential(nn.Conv2d(N*(L+1),N*(L+1),kernel_size=(C,1),groups=N*(L+1)),
                                nn.BatchNorm2d(N*(L+1)),
                                nn.ELU(),
                                nn.Dropout(dropout))
        self.tam = TemporalAttention((N*(L+1)))
        self.msem = MSEM((N*(L+1)),conv1_chans=MSEM_C1,conv2_chans=MSEM_C2)
        self.cblock = C_Block(T//2,(MSEM_C1+MSEM_C2)*3+(N*(L+1)),num_classes=num_classes)

    def forward(self,x):
        # print('Input:',x.shape)
        x = self.tc1(x)
        # print('TC-block:',x.shape)
        x = self.mha(x)
        # print('MHA:',x.shape)
        x = self.cc_block(x)
        # print('CC-block:',x.shape)
        x = self.tam(x)
        # print('TAM:',x.shape)
        x = self.msem(x)
        # print('MSEM:',x.shape)
        return self.cblock(x)