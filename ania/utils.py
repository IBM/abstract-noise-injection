import torch
import numpy as np


device = "cuda:0" if torch.cuda.is_available() else "cpu"

class StateLess(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

        self.delete_state()
    
    def delete_state(self):
        for name, _ in list(self.module.named_parameters()):
            parent, param_name = get_direct_parent(self.module, name)
            del parent._parameters[param_name]

    def forward(self, param_dict, *args, **kwargs):
        for name, param in param_dict.items():
            parent, param_name = get_direct_parent(self.module, name)
            #param.requires_grad_(True)
            setattr(parent, param_name, param)
        out = self.module.forward(*args, **kwargs)
        self.delete_state()
        return out

def map_torch(func, *inps, param=False):
    t = type(inps[0])
    if t in (dict, torch.nn.ParameterDict,torch.nn.modules.container.ParameterDict):
        if param:
            ret = torch.nn.ParameterDict()
        else:
            ret = {}
        for key in inps[0]:
            ret[key] = map_torch(func,*[inp[key] for inp in inps],param=param)
        return ret
    if t in (list,torch.nn.ParameterList,torch.nn.modules.container.ParameterList):
        if param:
            ret = torch.nn.ParameterList()
        else:
            ret = []
        for tup in zip(inps):
            ret.append(map_torch(func,*tup,param))
        return ret
    ret = func(*inps)
    if param:
        ret = torch.nn.Parameter(ret)
    return ret

def flatten_torch(inp):
    if type(inp) in (dict, torch.nn.ParameterDict,torch.nn.modules.container.ParameterDict):
        for key, val in inp.items():
            for kkey, vval in flatten_torch(val):
                if kkey is None:
                    kkey = key
                else:
                    kkey = f"{key}.{kkey}"
                yield kkey, vval
    elif type(inp) in (list,torch.nn.ParameterList,torch.nn.modules.container.ParameterList):
        for i, elem in enumerate(inp):
            for key, val in flatten_torch(elem):
                if key is None:
                    key = str(i)
                else:
                    f"{i}.{key}"
                yield key, val
    else:
        yield None, inp

def get_direct_parent(root, name):
    m = root
    path = name.split(".")
    for n in path[:-1]:
        if n.isdigit():
            m = m[int(n)]
            continue
        m = getattr(m,n)
    return m, path[-1]


def lexicographical_backward(param_dict, losses): 
    grads = None
    steps = 0
    for loss in losses:
        for key, param in param_dict.items():
            param.grad = None
        loss.backward()
        steps +=1

        new_grads = {name:param.grad for name,param in param_dict.items()}
        if grads is None:
            grads = new_grads
            continue
        
        for key in grads:
            grads[key] = grads[key] + new_grads[key] - grads[key] * np.dot(new_grads[key].reshape((-1,)), grads[key].reshape((-1,))) / np.norm(grads[key])**2
    return grads, steps