import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from src.pytorch_models.FTDNNPronscorer import *

def get_layer_type(is_tdnnf):
	layer_type = 'tdnn'
	if is_tdnnf:
		layer_type = 'tdnnf'
	return layer_type

def extract_batchnorm_count(string, layer_number, is_tdnnf = True, component_name='', dim='1536'):
	if component_name == '':
		layer_type = get_layer_type(is_tdnnf)
		component_name = layer_type + str(layer_number) + '.batchnorm'
	string = string.replace('<ComponentName> '+component_name+' <BatchNormComponent> <Dim> '+dim+' <BlockDim> '+dim+' <Epsilon> 0.001 <TargetRms> 1 <TestMode> F <Count> ', '')
	count = ''
	i = 0
	while string[i] != ' ':
		count = count + string[i]
		i = i+1
	return count

def read_linear_params(file):
	global line
	linear_params = []
	while ']' not in line:
		line = file.readline()
		line_array = np.fromstring(line.replace(']',''), dtype=float, sep=' ')
		linear_params.append(line_array)
	linear_params = np.array(linear_params)
	return linear_params

def read_vector(file, prefix, advance):
	global line
	if advance: 
		line = file.readline()
	line = line.replace(prefix, '')
	line = line.replace(']', '')
	if len(line) <= 1:
		return np.array([])
	vector = np.fromstring(line, dtype=float, sep=' ')
	return vector

def read_bias(file):
	return read_vector(file, '<BiasParams>  [ ', True)

def read_value_avg(file, layer_number, is_tdnnf = True):
	layer_type = get_layer_type(is_tdnnf)
	return read_vector(file, '<ComponentName> '+ layer_type + str(layer_number) +'.relu <RectifiedLinearComponent> <Dim> 1536 <ValueAvg>  [ ', False)

def read_deriv_avg(file):
	return read_vector(file, '<DerivAvg>  [ ', False)

def read_stats_mean(file, layer_number, is_tdnnf = True, component_name='', dim='1536'):
	if component_name == '':
		layer_type = get_layer_type(is_tdnnf)
		component_name = layer_type + str(layer_number) + '.batchnorm'
	count = extract_batchnorm_count(line,layer_number, is_tdnnf=is_tdnnf, component_name=component_name, dim=dim)
	return read_vector(file, '<ComponentName> '+ component_name +' <BatchNormComponent> <Dim> '+dim+' <BlockDim> '+dim+' <Epsilon> 0.001 <TargetRms> 1 <TestMode> F <Count> '+ count +' <StatsMean>  [ ', False)

def read_stats_var(file):
	return read_vector(file, '<StatsVar>  [ ', True)

def read_affine_component(file):
	params_dict = {}
	params_dict['linear_params'] = read_linear_params(file)
	params_dict['bias'] = read_bias(file)
	return params_dict

def read_batchnorm_component(file, layer_number, is_tdnnf=True, component_name='', dim='1536'):
	params_dict = {}
	params_dict['stats_mean'] = read_stats_mean(file, layer_number, is_tdnnf=is_tdnnf, component_name=component_name, dim=dim)
	params_dict['stats_var'] = read_stats_var(file)
	return params_dict

def read_relu_component(file, layer_number, is_tdnnf=True):
	params_dict = {}
	params_dict['value_avg'] = read_value_avg(file, layer_number, is_tdnnf=is_tdnnf)
	line = chain_file.readline()
	params_dict['deriv_avg'] = read_deriv_avg(file)
	return params_dict


def main(config_dict):
	global line
	global chain_file

	chain_file  = config_dict["libri-chain-txt-path"]
	output_path = config_dict["finetune-model-path"]
	phone_count = config_dict["phone-count"]
	batchnorm   = config_dict["batchnorm"] 
	seed        = config_dict["seed"]

	chain_file = open(chain_file, 'r') 
	 
	components = {}
	finished = False

	while not finished: 
	  
		line = chain_file.readline() 

		if line == '</Nnet3> ':
			finished = True

		if '<ComponentName> lda' in line:
			components['lda'] = read_affine_component(chain_file)

		if '<ComponentName> tdnn1.affine' in line:
			#If necessary, parse parameters such as learning rate here
			components['tdnn1.affine'] = read_affine_component(chain_file)

		if '<ComponentName> tdnn1.relu' in line:
			components['tdnn1.relu'] = read_relu_component(chain_file, 1, is_tdnnf=False)

		if '<ComponentName> tdnn1.batchnorm' in line:
			components['tdnn1.batchnorm'] = read_batchnorm_component(chain_file, 1, is_tdnnf=False)

		#No tdnn1.dropout yet


		#Reads parameters for layers 2-17
		for layer_number in range(2, 18):
			layer_name = 'tdnnf'+str(layer_number)

			if '<ComponentName> ' + layer_name + '.linear' in line:
				line = chain_file.readline() #Skip unnecessary line
				components[layer_name + '.linear'] = read_affine_component(chain_file)

			if '<ComponentName> ' + layer_name + '.affine' in line:
				line = chain_file.readline() #Skip unnecessary line
				components[layer_name + '.affine'] = read_affine_component(chain_file)

			if '<ComponentName> ' + layer_name + '.relu' in line:
				components[layer_name + '.relu'] = read_relu_component(chain_file, layer_number)

			if '<ComponentName> ' + layer_name + '.batchnorm' in line:
				components[layer_name + '.batchnorm'] = read_batchnorm_component(chain_file, layer_number)

			#No tdnnfx.dropout yet
			#No tdnnfx.noop yet
		
		if '<ComponentName> prefinal-l' in line:
			components['prefinal-l'] = {}
			components['prefinal-l']['linear_params'] = read_linear_params(chain_file)
		
		# The code used to parse the chain head parameters has been removed because a new layer will be used instead.


	phone_count = int(phone_count)

	ftdnn = FTDNNPronscorer(out_dim=phone_count, batchnorm=batchnorm)

	model_state_dict = {}

	model_state_dict['ftdnn.layer01.lda.weight']      = torch.from_numpy(components['lda']['linear_params'])
	model_state_dict['ftdnn.layer01.lda.bias']        = torch.from_numpy(components['lda']['bias'])
	model_state_dict['ftdnn.layer01.kernel.weight']   = torch.from_numpy(components['tdnn1.affine']['linear_params'])
	model_state_dict['ftdnn.layer01.kernel.bias']     = torch.from_numpy(components['tdnn1.affine']['bias'])
	model_state_dict['ftdnn.layer01.bn.running_mean'] = torch.from_numpy(components['tdnn1.batchnorm']['stats_mean'])
	model_state_dict['ftdnn.layer01.bn.running_var']  = torch.from_numpy(components['tdnn1.batchnorm']['stats_var'])



	for layer_number in range(2, 18):
		model_state_dict['ftdnn.layer'+ str("{:02d}".format(layer_number)) +'.sorth.weight'] = torch.from_numpy(components['tdnnf'+ str(layer_number) +'.linear']['linear_params'])
		model_state_dict['ftdnn.layer'+ str("{:02d}".format(layer_number)) +'.affine.weight'] = torch.from_numpy(components['tdnnf'+ str(layer_number) +'.affine']['linear_params'])
		model_state_dict['ftdnn.layer'+ str("{:02d}".format(layer_number)) +'.affine.bias'] = torch.from_numpy(components['tdnnf'+ str(layer_number) +'.affine']['bias'])
		model_state_dict['ftdnn.layer'+ str("{:02d}".format(layer_number)) +'.sorth.weight'] = torch.from_numpy(components['tdnnf'+ str(layer_number) +'.linear']['linear_params'])
		model_state_dict['ftdnn.layer'+ str("{:02d}".format(layer_number)) +'.bn.running_mean'] = torch.from_numpy(components['tdnnf'+ str(layer_number) +'.batchnorm']['stats_mean'])
		model_state_dict['ftdnn.layer'+ str("{:02d}".format(layer_number)) +'.bn.running_var'] = torch.from_numpy(components['tdnnf'+ str(layer_number) +'.batchnorm']['stats_var'])

	model_state_dict['ftdnn.layer18.weight'] = torch.from_numpy(components['prefinal-l']['linear_params'])

	
	#Add layer to finetune 
	torch.manual_seed(seed)
	model_state_dict['output_layer.linear.weight']   = torch.randn([phone_count, 256])
	model_state_dict['output_layer.linear.bias']     = torch.randn([phone_count])
	if batchnorm in ["final", "last", "firstlast"]:
		model_state_dict['output_layer.bn.running_mean'] = torch.zeros(256)
		model_state_dict['output_layer.bn.running_var']  = torch.ones(256)

	torch.nn.init.xavier_uniform(ftdnn.output_layer.linear.weight)

	for name, param in ftdnn.named_parameters():
		print (name, param.shape)

	chain_file.close() 

	state_dict = {}
	state_dict['model_state_dict'] = model_state_dict

	ftdnn.load_state_dict(model_state_dict)

	torch.save(state_dict, output_path)
