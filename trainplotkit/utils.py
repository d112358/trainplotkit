from typing import Mapping
import torch
from torch.utils.data import Dataset

# Device utilities (def_device, to_device, to_cpu) from miniai (https://github.com/fastai/course22p2/tree/master/miniai)
def_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

def to_device(x, device=def_device):
    if isinstance(x, torch.Tensor): return x.to(device)
    if isinstance(x, Mapping): return {k:v.to(device) for k,v in x.items()}
    return type(x)(to_device(o, device) for o in x)

def to_cpu(x):
    if isinstance(x, Mapping): return {k:to_cpu(v) for k,v in x.items()}
    if isinstance(x, list): return [to_cpu(o) for o in x]
    if isinstance(x, tuple): return tuple(to_cpu(list(x)))
    res = x.detach().cpu()
    return res.float() if res.dtype==torch.float16 else res

class AxisRange():
    """
    Convenience class to simplify keeping ranges updated with plot contents
    """
    def __init__(self, min_x=0, max_x=1, min_y=0, max_y=1):
        self.min_x, self.max_x, self.min_y, self.max_y = min_x, max_x, min_y, max_y

    def x_range(self): return [self.min_x, self.max_x]
    def y_range(self): return [self.min_y, self.max_y]

    def update(self, x_values, y_values):
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)

        range_changed = False
        if min_x < self.min_x: self.min_x = min_x; range_changed = True
        if max_x > self.max_x: self.max_x = max_x; range_changed = True
        if min_y < self.min_y: self.min_y = min_y; range_changed = True
        if max_y > self.max_y: self.max_y = max_y; range_changed = True
        return range_changed

class InMemDataset(Dataset):
    """
    A dataset object for data that fits entirely into memory
    """
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
