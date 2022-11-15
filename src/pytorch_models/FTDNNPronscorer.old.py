from this import d
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from src.pytorch_models.FTDNN import FTDNN



def summarize_outputs_per_phone_logpost(outputs, batch_target_phones, batch_cum_matrix): 
    
    zeros = torch.zeros(outputs.shape[0], outputs.shape[1], outputs.shape[2], device='cuda:0')
    frame_log_post = -torch.logaddexp(zeros,-outputs) #log p en cada posicion
    phone_log_post = summarize_outputs_per_phone(frame_log_post, batch_target_phones, batch_cum_matrix)
    eps = 1e-12
    phone_logits = phone_log_post - torch.log((-torch.special.expm1(phone_log_post))+eps)


    return phone_logits

def summarize_outputs_per_phone(outputs, batch_target_phones, batch_cum_matrix): 
    # poner el summarize acá adentro y repetir el if 
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
        self.output_layer = OutputLayer(257, out_dim, use_bn=use_final_bn)
        
        
    def forward(self, x, loss_per_phone, summarize, eval, batch_target_phones, batch_cum_matrix):
        '''
        Input must be (batch_size, seq_len, in_dim)
        '''
        x = self.ftdnn(x)
        # nuevo x que sea la concatencacion de x con batch_target phones
        # cambiar el output layer a out_dim=1 
        # el forward da una proba por frame
        # las etiquetas son el equivalente a batch target phones colapsado porque tengo una etiqueta por frame. 
        # el resumen se hace antes de la loss. 
        x = self.output_layer(x)

        # llamar score_per_phone or eval y hago lo mismo en los dos casos
        if loss_per_phone:

            if summarize == "m_logpost":
                x = summarize_outputs_per_phone_logpost(x, batch_target_phones, batch_cum_matrix)
            else: 
                x = summarize_outputs_per_phone(x, batch_target_phones, batch_cum_matrix)
        
        if eval:
            # Otra manera de evaluar - MAX
            if summarize == "m_logpost":
                # se puede cambiar el torch zeros
                x = -torch.logaddexp(torch.zeros(x.shape[0], x.shape[1], x.shape[2]),-x)
            x = summarize_outputs_per_phone(x, batch_target_phones, batch_cum_matrix)
            x = torch.sum(x, dim=2)

        return x
