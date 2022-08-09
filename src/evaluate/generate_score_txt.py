# mod
from src.pytorch_models.FTDNNPronscorer import FTDNNPronscorer

import torch
import torch.optim as optim

from src.utils.finetuning_utils import *
from src.train.dataset import *
from torch.optim.swa_utils import AveragedModel
from IPython import embed
import argparse

def removeSymbols(str, symbols):
    for symbol in symbols:
        str = str.replace(symbol,'')
    return str

def generate_scores_for_testset(model, testloader):
    print('Generating scores for testset')
    
    evaluation = True
    loss_per_phone = False 
    #summarize = False
  
    scores = {}
    for i, batch in enumerate(testloader, 0):       
        print('Batch ' + str(i+1) + '/' + str(len(testloader)))
        
        logids      = unpack_logids_from_batch(batch)
        features    = unpack_features_from_batch(batch)
        batch_target_phones = unpack_ids_from_batch(batch)
        batch_transcripts =  unpack_transcriptions_from_batch(batch)
        batch_cum_matrix = unpack_cum_matrix_from_batch(batch)

    
        outputs = (-1) * model(features, loss_per_phone, evaluation, 
                                batch_target_phones, batch_cum_matrix)

    
        
        for j, logid in enumerate(logids, 0):
          
            p = batch_transcripts[j].detach().numpy()
            o = outputs[j].detach().numpy()
            scores[logid] = [o[o!=0], p[p!=0]]
        
    return scores


def log_sample_scores_to_txt(logid, sample_scores, score_log_fh):
    score_log_fh.write(logid + ' ')
    scores = sample_scores[0]
    phone_number = sample_scores[1]
    
    for i, score in enumerate(scores):
        score_log_fh.write( '[ ' + str(phone_number[i]) + ' ' + str(score)  + ' ] ')
    score_log_fh.write('\n')

def log_testset_scores_to_txt(scores, score_log_fh):
    print('Writing scores to .txt')
    for logid, sample_score in scores.items():
        log_sample_scores_to_txt(logid, sample_score, score_log_fh)

def main(config_dict):

    state_dict_dir      = config_dict['state-dict-dir']
    model_name          = config_dict['model-name']
    sample_list         = config_dict['utterance-list-path']
    phone_list_path     = config_dict['phones-list-path']
    labels_dir          = config_dict['auto-labels-dir-path']
    gop_txt_dir         = config_dict['gop-scores-dir']
    gop_txt_name        = config_dict['gop-txt-name']
    batch_size          = config_dict['batch-size']
    features_path       = config_dict['features-path']
    conf_path           = config_dict['features-conf-path']
    device_name         = config_dict['device']
    batchnorm           = config_dict['batchnorm']


    testset = EpaDB(sample_list, phone_list_path, labels_dir, features_path, conf_path)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=0, collate_fn=collate_fn_padd)

    phone_count = testset.phone_count()

    #Get pronscoring model to test
    model = FTDNNPronscorer(out_dim=phone_count, device_name=device_name, batchnorm=batchnorm)
    if model_name.split("_")[-1] == "swa":
        model = AveragedModel(model)
    
    model.eval()
    state_dict = torch.load(state_dict_dir + '/' + model_name + '.pth')
    model.load_state_dict(state_dict['model_state_dict'])

    phone_dict = testset._phone_sym2int_dict
    
    scores = generate_scores_for_testset(model, testloader)
    score_log_fh = open(gop_txt_dir+ '/' + gop_txt_name, 'w+')
    log_testset_scores_to_txt(scores, score_log_fh)

