import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from src.pytorch_models.FTDNN import FTDNN



def summarize_outputs_per_phone_gop(outputs, batch_target_phones, batch_cum_matrix): 
    outputs = -torch.logaddexp(torch.zeros(outputs.shape[0], outputs.shape[1], outputs.shape[2]),-outputs)
    by_phone_outputs = summarize_outputs_per_phone(outputs, batch_target_phones, batch_cum_matrix)
    p_outputs = torch.exp(by_phone_outputs) 
    odds_denominator = torch.ones(p_outputs.shape[0], p_outputs.shape[1], p_outputs.shape[2])-p_outputs
    logodds_outputs = torch.div(p_outputs, odds_denominator)
    return logodds_outputs

def summarize_outputs_per_phone(outputs, batch_target_phones, batch_cum_matrix): 
    
    masked_outputs = outputs*abs(batch_target_phones)
    summarized_outputs = torch.matmul(batch_cum_matrix, masked_outputs)
    frame_counts = torch.matmul(batch_cum_matrix, batch_target_phones)
    frame_counts[frame_counts==0]=1
    by_phone_outputs = torch.div(summarized_outputs, frame_counts)

    return by_phone_outputs

class OutputLayer(nn.Module):

    def __init__(self, in_dim, out_dim, use_bn=False):

        super(OutputLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bn = use_bn

        if use_bn:
            self.bn = nn.BatchNorm1d(self.in_dim, affine=False)
        self.linear = nn.Linear(self.in_dim, self.out_dim, bias=True) 
        self.nl = nn.Sigmoid()

    def forward(self, x):
        if self.use_bn:
            x = x.transpose(1,2)
            x =self.bn(x).transpose(1,2)
        x = self.linear(x)
        return x

class FTDNNPronscorer(nn.Module):

    def __init__(self, out_dim=40, batchnorm=None, dropout_p=0, device_name='cpu'):

        super(FTDNNPronscorer, self).__init__()

        use_final_bn = False
        if batchnorm in ["final", "last", "firstlast"]:
            use_final_bn=True
        
        self.ftdnn        = FTDNN(batchnorm=batchnorm, dropout_p=dropout_p, device_name=device_name)
        self.output_layer = OutputLayer(256, out_dim, use_bn=use_final_bn)
        
    def forward(self, x, loss_per_phone, summarize, eval, batch_target_phones, batch_cum_matrix):
        '''
        Input must be (batch_size, seq_len, in_dim)
        '''
        x = self.ftdnn(x)
        x = self.output_layer(x)
         

        if loss_per_phone:

            if summarize == 'gop':
                x = summarize_outputs_per_phone_gop(x, batch_target_phones, batch_cum_matrix)
            else: 
                x = summarize_outputs_per_phone(x, batch_target_phones, batch_cum_matrix)
        
        if eval:
            #if summarize == 'gop':

            x = -torch.logaddexp(torch.zeros(x.shape[0], x.shape[1], x.shape[2]),-x)
            x = summarize_outputs_per_phone(x, batch_target_phones, batch_cum_matrix)
            x = torch.sum(x, dim=2)

        return x
