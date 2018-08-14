# PyTorchOT

Implements sinkhorn optimal transport algorithms in PyTorch. Currrently there are two versions of the Sinkhorn 
algorithm implemented: [the original](https://arxiv.org/pdf/1306.0895.pdf) and the [log-stabilized version](https://arxiv.org/pdf/1610.06519.pdf).

Example usage:
```
from ot_pytorch import sink

M = pairwise_distance_matrix()
dist = sink(M, reg=5, cuda=False)
```

Setting cuda=True enables cuda use.

The examples.py file contains two basic examples. 

Example 1: 

Let Z<sub>i</sub> ~ Uniform[0,1], and define the data X<sub>i</sub> = (0,Z<sub>i</sub>), Y<sub>i</sub> = (θ, Z<sub>i</sub>), for i=1,...,N and some parameter θ which is varied over [-1,1]. The true optimal transport distance is |θ|. The algorithm yields:

![alt text](https://github.com/rythei/PyTorchOT/blob/master/plots/uniform_example/uniform_example2.png)

