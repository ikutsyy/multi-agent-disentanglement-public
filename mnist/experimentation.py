import numpy as np
import probtorch
import torch
from matplotlib import pyplot as plt

size = 5
space = np.ones(size)/size

xs = np.arange(20)
ys = []

for temp in list(np.linspace(0.001,10,num=20)):
    q = probtorch.Trace()
    digits = q.concrete(probs=torch.tensor(space),
                                temperature=torch.tensor(temp),
                                value=None,
                                name='digits')

    print(np.asarray(digits))
    print(q['digits'])
    norm = np.linalg.norm(digits)
    ys.append((np.sqrt(size)*norm-1)/(np.sqrt(size)-1))

#plt.plot(xs,ys)
#plt.show()


