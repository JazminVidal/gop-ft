import os
import glob
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from IPython import embed


def unpack_logids_from_batch(batch):
    return [item['speaker_id'] + '_' + item['utterance_id'] for item in batch]

def unpack_features_from_batch(batch):
    return torch.stack([item['features'] for item in batch])

def unpack_pos_labels_from_batch(batch):
    return torch.stack([item['pos_labels'] for item in batch])

def unpack_labels_from_batch(batch):
    return torch.stack([item['labels'] for item in batch])

def unpack_neg_labels_from_batch(batch):
    return torch.stack([item['neg_labels'] for item in batch])

def unpack_ids_from_batch(batch):
    return torch.stack([item['ids'] for item in batch])

def unpack_phone_times_from_batch(batch):
    return [item['phone_times'] for item in batch]

def unpack_transcriptions_from_batch(batch):
    return  torch.stack([item['transcription'] for item in batch])

def unpack_durations_from_batch(batch):
    return  torch.stack([item['durations'] for item in batch])

def unpack_cum_matrix_from_batch(batch):
    return torch.stack([item['cum_matrix'] for item in batch])

def re_pad_index(index):
    last_i = 0
    non_zero = True
    for i in range(len(index)-1):
        if index[i+1]==0 and non_zero == True:
            last_i = index[i]
            non_zero = False
    index = np.where(index == 0, last_i, index)
    return(index)

def unpack_transitions_from_batch(batch):
    
    all_indexes = []
    for i, item in enumerate(batch):
        index = item['transition_indexes'].numpy()
        index = re_pad_index(index)
        batch_position = np.repeat(i,len(index))
        trans_index = np.array([batch_position, index])
        all_indexes.append(trans_index)
    return np.stack([ind for ind in all_indexes], axis=1)

def get_max_row_and_col_length(list_of_tensors):
    mr = []
    mc = []
    for t in list_of_tensors: 
        mr.append(t.shape[0])
        mc.append(t.shape[1])
    return np.max(mr), np.max(mc)

def ConstantPad2d(list_of_tensors, mr, mc):
    padded = []
    for t in list_of_tensors: 
        r0 = t.shape[0]
        c0 = t.shape[1]
        pad = nn.ConstantPad2d((0, mc-c0, 0, mr-r0), 0)
        padded.append(pad(t))
    return padded    



def collate_fn_padd(batch):
    '''
    Padds batch of variable length (both features and labels)
    '''
    ## padd
    batch_features   = [item['features'] for item in batch]
    batch_features   = torch.nn.utils.rnn.pad_sequence(batch_features, batch_first=True)
    batch_pos_labels = [item['pos_labels'] for item in batch]
    batch_pos_labels = torch.nn.utils.rnn.pad_sequence(batch_pos_labels, batch_first=True, padding_value=-1)
    batch_neg_labels = [item['neg_labels'] for item in batch]
    batch_neg_labels = torch.nn.utils.rnn.pad_sequence(batch_neg_labels, batch_first=True, padding_value=-1)
    batch_labels     = [item['labels'] for item in batch]
    batch_labels     = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=0)
    batch_ids        = [item['ids'] for item in batch]
    batch_ids        = torch.nn.utils.rnn.pad_sequence(batch_ids, batch_first=True, padding_value=0)
    batch_indexes    = [item['transition_indexes'] for item in batch]
    batch_indexes    = torch.nn.utils.rnn.pad_sequence(batch_indexes, batch_first=True, padding_value=0)
    batch_transcripts = [item['transcription'] for item in batch]
    batch_transcripts = torch.nn.utils.rnn.pad_sequence(batch_transcripts, batch_first=True, padding_value=0)
    batch_durations = [item['durations'] for item in batch]
    batch_durations = torch.nn.utils.rnn.pad_sequence(batch_durations, batch_first=True, padding_value=0)
    batch_cum_matrix  = [item['cum_matrix'] for item in batch]
    mr, mc = get_max_row_and_col_length(batch_cum_matrix)
    batch_cum_matrix = ConstantPad2d(batch_cum_matrix, mr, mc)


    for i in range(len(batch)):
        batch[i]['features']   = batch_features[i]
        batch[i]['pos_labels'] = batch_pos_labels[i]
        batch[i]['neg_labels'] = batch_neg_labels[i]
        batch[i]['labels']     = batch_labels[i]
        batch[i]['ids']        = batch_ids[i]
        batch[i]['transition_indexes'] = batch_indexes[i]
        batch[i]['transcription'] = batch_transcripts[i]
        batch[i]['durations'] = batch_durations[i]
        batch[i]['cum_matrix'] = batch_cum_matrix[i]
    return batch

#phone_sym2int_dict:  Dictionary mapping phone symbol to integer given a phone list path
#phone_int2sym_dict:  Dictionary mapping phone integer to symbol given a phone list path
#phone_int2node_dict: Dictionary mapping phone symbol to the index of the node in the networks's output layer
#NOTE: The node number in the output layer is not the same as the phone number, as some phones will not be scored
def get_phone_dictionaries(phone_list_path):
    #Open file that contains list of pure phones
    phones_list_fh = open(phone_list_path, "r")

    phone_sym2int_dict  = {}
    phone_int2sym_dict  = {}
    phone_int2node_dict = {}
    current_node_index  = 0
    #Populate the dictionaries
    for line in phones_list_fh.readlines():
        line = line.split()
        phone_symbol = line[0]
        phone_number = int(line[1])
        use_phone    = bool(int(line[2]))
        if use_phone:
            phone_sym2int_dict[phone_symbol]    = phone_number
            phone_int2sym_dict[phone_number]    = phone_symbol
            phone_int2node_dict[phone_number]   = current_node_index
            current_node_index += 1

    return phone_sym2int_dict, phone_int2sym_dict, phone_int2node_dict

#The model outputs a score for each phone in each frame. This function extracts only the relevant scores,
#i.e the scores for the canonic phone in each frame based on the annotations.
#If a frame has no canonic phone (silence frame), the score is set to 0.
def get_scores_for_canonic_phones(outputs, labels):
    #Generate mask based on non-zero labels
    outputs_mask = torch.abs(labels)
    #Mask outputs and sum over phones to get a single value for the relevant phone in each frame
    outputs = outputs * outputs_mask
    outputs = torch.sum(outputs, dim=2)
    return outputs

#This function returns the non-zero relevant scores and 0/1 labels to calculate loss
def get_outputs_and_labels_for_loss(outputs, labels):
    outputs = get_scores_for_canonic_phones(outputs, labels)
    #Sum over phones to keep relevant label for each frame
    labels = torch.sum(labels, dim=2)
    #Remove labels == 0 (silence frames) in both labels and outputs
    outputs = outputs[labels != 0]
    labels = labels[labels != 0]
    #Turn 1s into 0s and -1s into 1s to pass the labels to loss_fn
    labels = labels - 1
    labels = torch.abs(labels / 2)    
    return outputs, labels


#Returns the canonic phone number at a given frame in the labels 
def get_phone_number_at_frame(labels, frame):
    #try: 
    res = labels[frame].nonzero().item()
    #except ValueError as e:
    #	embed()
    return res + 3 #This +3 is here because librispeech phones-pure starts with eps, sil, spn and these should be skipped 

#Collapses multiple frame level scores using sum or mean  
def get_phone_score_from_frame_scores(frame_level_scores, start_time, end_time, method):
    if   method == 'sum':
        return torch.sum(frame_level_scores[start_time:end_time]).item()
    elif method == 'mean':
        return torch.mean(frame_level_scores[start_time:end_time]).item()
    else:
        raise Exception('Unsupported frame score collapse method ' + method)


#This function takes a dataloader and writes the logids and paths of all samples
#in the dataloader to directory/filename. The generated sample list is used to train/test 
#the model for each fold. 
def generate_sample_list(dataloader, epa_root_path, directory, filename):
    sample_list_fh = open(directory + '/' + filename, 'w+')
    for i, data in enumerate(dataloader, 0):
        logids = unpack_logids_from_batch(data)
        for logid in logids:
            spkr_id = logid.split('_')[0]
            sample_path = epa_root_path + '/' + spkr_id + '/waveforms/' + logid + '.wav'
            sample_list_fh.write(logid + ' ' + sample_path + '\n')

#This function takes the path to an utterance list and returns a list of logids 
#and a dictionary mapping speaker ids to logid lists 
def generate_fileid_list_and_spkr2logid_dict(sample_list_path):
    file_id_list = []
    logids_by_speaker = {}
    sample_list_fh = open(sample_list_path, "r")
    for line in sample_list_fh.readlines():
        line = line.split()
        logid = line[0]
        speaker_id = logid.split('_')[0]
        file_id_list.append(logid)
        if speaker_id in logids_by_speaker:
            logids_by_speaker[speaker_id].append(logid)
        else:
            logids_by_speaker[speaker_id] = [logid]
    return file_id_list, logids_by_speaker

#This function takes the path to an utterance list and returns a list of speaker ids
def get_speaker_list_from_utterance_list_file(sample_path):
    speaker_list = []
    sample_list_fh = open(sample_path, "r")
    for line in sample_list_fh.readlines():
        line = line.split()
        logid = line[0]
        speaker_id = logid.split('_')[0]
        if speaker_id not in speaker_list:
            speaker_list.append(speaker_id)
    return speaker_list