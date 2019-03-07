# tcop-pytorch

This is my collection of CUDA custom operators for PyTorch. It currently contains only `MaskedSoftmax`.

## Requirements

I tested this operator in this environment:

- PyTorch 1.0
- CUDA 9.2

## How to Install

```
$ python setup.py install
```

## How to Test
```
$ python setup.py test
```

## How to Use

### MaksedSoftmax

```MaskedSoftmax.apply(input, mask, scale)```

You can find details about this operator in [my blog post](https://tunz.kr/post/5).

```python
import torch
from tcop.masked_softmax import MaskedSoftmax

x = torch.tensor([[0.3, 0.2, 0.1],
                  [0.3, 0.4, 0.5]]).cuda()
x = x.view(1, 1, 2, 3)  # [batch_size, head_size, q, k]

mask = torch.tensor([2, 1], dtype=torch.int32).cuda()
mask = mask.view(1, 2)  # [batch_size, q]

scale = 0.5
MaskedSoftmax.apply(x, mask, scale)
# tensor([[[[0.5125, 0.4875, 0.0000],
#           [1.0000, 0.0000, 0.0000]]]], device='cuda:0')
```
