# PyTorchOT

This package implements sinkhorn ptimal transport algorithms in PyTorch. Currrently there are two versions of the Sinkhorn 
algorithm implemented: [the original](https://arxiv.org/pdf/1306.0895.pdf) and the [log-stabilized version](https://arxiv.org/pdf/1610.06519.pdf).

Example usage:
```
from ot_pytorch import sink

M = pairwise_distance_matrix()
dist = sink(M, reg=5, cuda=False)
```

