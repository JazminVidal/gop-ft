from this import d
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from src.pytorch_models.FTDNN import FTDNN


def summarize_outputs_per_phone(outputs, batch_target_phones, batch_cum_matrix, summarize, eval): 
    
    if summarize == "m_logpost":
        # Calcula log posteriors por frame
       
        zeros = torch.zeros(outputs.shape[0], outputs.shape[1], outputs.shape[2], device='cuda:0')
        if eval:
            zeros = zeros.cpu()
        outputs = -torch.logaddexp(zeros,-outputs) #log p en cada posicion
    
    masked_outputs = outputs*abs(batch_target_phones)

    if summarize == "min":
        M = torch.exp(masked_outputs)
        M_masked = M*abs(batch_target_phones)
        N = torch.matmul(batch_cum_matrix, M_masked)
        N_vec = torch.sum(N, dim=2)
        xpnd_N_vec = N_vec.unsqueeze(2).repeat(1,1,batch_cum_matrix.shape[2])
        cumm_N = batch_cum_matrix*xpnd_N_vec
        cumm_N_vec = torch.sum(cumm_N, dim=1)
        xpnd_cumm_N_vec = cumm_N_vec.unsqueeze(2).repeat(1,1,39)
        xpnd_cumm_N_vec[xpnd_cumm_N_vec==0]=1
        S = torch.div(M_masked,xpnd_cumm_N_vec)
        masked_outputs = 1-(S*masked_outputs) # para tener el mínimo.
        by_phone_outputs = torch.matmul(batch_cum_matrix, masked_outputs)


    if summarize != "min":
        
        summarized_outputs = torch.matmul(batch_cum_matrix, masked_outputs)
        frame_counts = torch.matmul(batch_cum_matrix, batch_target_phones)
        frame_counts[frame_counts==0]=1
        by_phone_outputs = torch.div(summarized_outputs, frame_counts)

    
    if summarize == "m_logpost":
       
        # Volves a logits - Esto en el caso de evaluación a mi no me cierra, a Luciana si. 
        # Lo tengo que ver de nuevo. 
        eps = 1e-12
        logit_denom = torch.log((-torch.special.expm1(by_phone_outputs))+eps)
        logit_denom_mask = logit_denom*abs((torch.sign(by_phone_outputs)))
        by_phone_outputs = by_phone_outputs - logit_denom_mask

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
        # nuevo x que sea la concatencacion de x con batch_target phones
        # cambiar el output layer a out_dim=1 
        # el forward da una proba por frame
        # las etiquetas son el equivalente a batch target phones colapsado porque tengo una etiqueta por frame. 
        # el resumen se hace antes de la loss. 
        x = self.output_layer(x)

        # se puede llamar score_per_phone 
        if loss_per_phone or eval:
            x = summarize_outputs_per_phone(x, batch_target_phones, batch_cum_matrix, summarize, eval)
        
        if eval:
            # Otra manera de evaluar - MIN
            x = torch.sum(x, dim=2)

        return x
