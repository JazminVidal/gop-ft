import yaml
from src.utils.run_utils import get_experiment_directory, get_run_name, add_data_keys_to_config_dict, get_phone_count

def add_gop_and_exp_common_keys(config_dict, config_yaml, use_heldout):
        config_dict["experiment-dir-path"] 	 = get_experiment_directory(config_yaml, use_heldout=use_heldout)
        config_dict["run-name"] 			 = get_run_name(config_yaml, use_heldout=use_heldout)
        config_dict["gop-scores-dir"] 		 = config_dict["experiment-dir-path"] 	  + "gop_scores/"	
        config_dict["eval-dir"] 			 = config_dict["experiment-dir-path"] 	  + "eval/"
        config_dict["held-out"]              = use_heldout
        config_dict["seed"]                  = 42
        return config_dict

class DataprepConfig():
    def __init__(self, config_yaml):
        config_fh   = open(config_yaml, "r")
        config_dict = yaml.safe_load(config_fh)

        config_dict = add_data_keys_to_config_dict(config_dict, "dataprep")

        self.config_dict = config_dict
    
class ExperimentConfig():
    def __init__(self, config_yaml, use_heldout, device_name):
        config_fh   = open(config_yaml, "r")
        config_dict = yaml.safe_load(config_fh)

        config_dict = add_data_keys_to_config_dict(config_dict, "exp")

        config_dict = add_gop_and_exp_common_keys(config_dict, config_yaml, use_heldout)

        config_dict["device"]                = device_name
        config_dict["phone-count"]           = get_phone_count(config_dict["phones-list-path"])
        config_dict["state-dict-dir"] 		 = config_dict["experiment-dir-path"] 	  + "state_dicts/"
        config_dict["test-sample-list-dir"]  = config_dict["experiment-dir-path"] 	  + "test_sample_lists/"
        config_dict["train-sample-list-dir"] = config_dict["experiment-dir-path"] 	  + "train_sample_lists/"

        if not use_heldout:
            config_dict["full-gop-score-path"] = config_dict["gop-scores-dir"] + "gop-all-folds.txt"

	    #If only one layer will be trained or finetune model path is not defined, make finetune model path relative to experiment dir
        if config_dict["layers"] == 1 or "finetune-model-path" not in config_dict:
            config_dict["finetune-model-path"]   = config_dict["experiment-dir-path"] + "/model_finetuning_kaldi.pt"

        self.config_dict = config_dict

class GopConfig():
    def __init__(self, config_yaml, use_heldout):
        config_fh   = open(config_yaml, "r")
        config_dict = yaml.safe_load(config_fh)

        config_dict = add_data_keys_to_config_dict(config_dict, "gop")

        config_dict = add_gop_and_exp_common_keys(config_dict, config_yaml, use_heldout)

        config_dict["eval-filename"]       = "data_for_eval.pickle"
        config_dict["full-gop-score-path"] = config_dict["gop-scores-dir"] + "gop.txt"

        if use_heldout:
            config_dict["utterance-list-path"] = config_dict["test-list-path"]
        else:
            config_dict["utterance-list-path"] = config_dict["train-list-path"]
        self.config_dict = config_dict


class AppConfig():
    def __init__(self, config_yaml, spkr_wav_list, wavs_list, name_set):
        config_fh   = open(config_yaml, "r")
        config_dict = yaml.safe_load(config_fh)

        data_path                            = config_dict["output-dir"]
        config_dict["alignments-dir-path"]   = data_path + "alignments/"
        config_dict["alignments-path"]       = config_dict["alignments-dir-path"] + "align_output_"+name_set
        config_dict["loglikes-path"]         = config_dict["alignments-dir-path"] + "loglikes_"+name_set+".ark"
        config_dict["features-conf-path"]    = data_path + "features/conf"
        config_dict["features-path"]         = data_path + "features/"+name_set
        config_dict["auto-labels-dir-path"]  = data_path + "kaldi_labels/"
        config_dict["train-list-path"]       = data_path + "/" + spkr_wav_list
        config_dict["test-list-path"]        = data_path + "/" + spkr_wav_list
        config_dict["utterance-list-path"]   = config_dict["test-list-path"]
        config_dict["wavs_list"]             = data_path + "/" + wavs_list
        config_dict["name_set"]              = name_set
        config_dict["gop-scores-dir"]        = data_path    + "gop_scores/"   
        config_dict["phone-count"]           = get_phone_count(config_dict["phones-list-path"])
        config_dict["state-dict-dir"]        = "pytorch_models/"
        config_dict["finetune-model-path"]   = config_dict["pronscoring-model-path"]  
        config_dict['batchnorm']             = "last"
        config_dict["ref-labels-dir-path"]   = config_dict["data-root-path"]
        config_dict["model-name"]            = config_dict["pronscoring-model-path"].split("/")[-1] 
        config_dict["utterance-list-path"]   = config_dict["test-list-path"]
        config_dict["gop-txt-name"]          = 'gop-'+config_dict["model-name"]+'-'+name_set+'.txt'

        self.config_dict = config_dict