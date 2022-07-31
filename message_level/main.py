import os
import sys

from disentanglement.message_level.model_parameters import SAVE_NAME, weights
from disentanglement.message_level.train import get_models

if __name__ == '__main__':
    savedir = "saved_models"
    load = False
    skip_obs_agents = False
    allowed_chunks = None
    epochs = 10

    if "a" in sys.argv[1]:
        weights["G"] = 0
        modelname = "nofactorvae_pickled_model_" + SAVE_NAME + ".pkl"
        checkpoint_path = os.path.join("saved_models", "nofactorvae_checkpoints")
        training_data_name = "nofactorvae_training_data_" + SAVE_NAME + ".pkl"
    elif "b" in sys.argv[1]:
        modelname = "noobs_pickled_model_" + SAVE_NAME + ".pkl"
        checkpoint_path = os.path.join("saved_models", "noobs_checkpoints")
        training_data_name = "noobs_training_data_" + SAVE_NAME + ".pkl"
        skip_obs_agents = True
    elif "c" in sys.argv[1]:
        weights["G"] = 0
        modelname = "nothing_pickled_model_" + SAVE_NAME + ".pkl"
        checkpoint_path = os.path.join("saved_models", "nothing_checkpoints")
        training_data_name = "nothing_training_data_" + SAVE_NAME + ".pkl"
        skip_obs_agents = True
    elif "d" in sys.argv[1]:
        modelname = "partial_pickled_model_" + SAVE_NAME + ".pkl"
        checkpoint_path = os.path.join("saved_models", "partial_checkpoints")
        training_data_name = "partial_training_data_" + SAVE_NAME + ".pkl"
        allowed_chunks = [0]
    elif "e" in sys.argv[1]:
        weights["B"] = 0.001
        modelname = "lowbeta_pickled_model_" + SAVE_NAME + ".pkl"
        checkpoint_path = os.path.join("saved_models", "lowbeta_checkpoints")
        training_data_name = "lowbeta_training_data_" + SAVE_NAME + ".pkl"
    elif "f" in sys.argv[1]:
        modelname = "smalldecoder_pickled_model_" + SAVE_NAME + ".pkl"
        checkpoint_path = os.path.join("saved_models", "smalldecoder_checkpoints")
        training_data_name = "smalldecoder_training_data_" + SAVE_NAME + ".pkl"
    elif "g" in sys.argv[1]:
        weights["B"] = 0.001
        weights["A"] = 100
        modelname = "lbha_pickled_model_" + SAVE_NAME + ".pkl"
        checkpoint_path = os.path.join("saved_models", "lbha_checkpoints")
        training_data_name = "lbha_training_data_" + SAVE_NAME + ".pkl"

    else:
        modelname = "pickled_model_" + SAVE_NAME + ".pkl"
        checkpoint_path = os.path.join("saved_models", "checkpoints")
        training_data_name = "training_data_" + SAVE_NAME + ".pkl"

    print(modelname)
    get_models(savedir, modelname, training_data_name, load, epochs, checkpoint_path,skip_obs_agents=skip_obs_agents)
