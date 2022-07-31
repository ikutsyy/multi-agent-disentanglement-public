import pickle
import codecs
import sys

import disentanglement.message_level.train as agent_train
import disentanglement.timestep_level.timestep_train as timestep_train
import disentanglement.message_level.model_parameters
from disentanglement.message_level.utils import ifcuda, alltocuda, alltocpu

if __name__ == '__main__':
    modelname = "pickled_model_" + sys.argv[1] + ".pkl"
    trainingname = "training_data_" + sys.argv[1] + ".pkl"
    if "agent" in sys.argv[2]:
        savedir = "../message_level/saved_models"
        enc, dec, _, _, _, _ = agent_train.get_models(savedir, modelname, trainingname, load=True)
    else:
        savedir = "../timestep_level/saved_models"
        enc, dec, _, _, _, _ = timestep_train.get_models(savedir, modelname, trainingname, load=True)
    while True:
        p = ""
        while True:
            got = sys.stdin.readline()
            if len(got) == 1:
                break
            p += got

        do_encoder, values, batch_size = pickle.loads(codecs.decode(p.encode(), "base64"))
        if batch_size==None:
            batch_size = disentanglement.message_level.model_parameters.BATCH_SIZE

        if do_encoder:
            q = enc(ifcuda(values))
            result = {}
            for k in q._nodes.keys():
                result[k] = q[k].value.cpu().detach()
        else:
            alltocuda(values)
            result = dec(None, q=values,batch_size=batch_size)[1].cpu().detach()
            alltocpu(values)

        print("return"+str(codecs.encode(pickle.dumps(result)+b'\n', "base64")))
