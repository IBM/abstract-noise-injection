from re import T
import torch
from ania.base_regularizer import BaseRegularizer


from ania.utils import StateLess, get_direct_parent, map_torch

class MultiplyBernoulliNoise(torch.nn.Module):
    def __init__(self,p):
        super().__init__()
        self.p = p

        self.ignore = False
    
    def forward(self, inp):
        if self.ignore:
            return inp
        
        #lambda inp:torch.bernoulli(1 - self.p * torch.ones_like(inp)) * inp / (1 - self.p)
        
        return map_torch(lambda ii: torch.nn.functional.dropout(ii,self.p), inp)




class BernoulliLayer(torch.nn.Module):
    
    def __init__(self, module, noisy_parameters, noisy_activations, p) :
        super().__init__()

        self.noisy_parameters = noisy_parameters
        self.noisy_activations = noisy_activations
        sd = module.named_parameters()
        self.mu_params = torch.nn.ParameterDict(sd)
        if self.noisy_parameters:
            self.add_param_noise = MultiplyBernoulliNoise(p)
        if self.noisy_activations:
            self.add_output_noise = MultiplyBernoulliNoise(p)
        

        self.module = StateLess(module)
    
    def forward(self, *args, **kwargs):
        params = self.mu_params
        if self.noisy_parameters:
            params = self.add_param_noise(params)
        out = self.module(params, *args, **kwargs)
        if self.noisy_activations:
            out = self.add_output_noise(out)
        return out

class BernoulliReg(BaseRegularizer):
    
    def ignore_noise(self, ignore):
        for module in self.module.modules():
            if type(module) is MultiplyBernoulliNoise:
                module.ignore = ignore

    def set_noisy(
        self, 
        activation_types=[], 
        activation_names=[], 
        parameter_types=[], 
        parameter_names=[], 
        replace_types=[], 
        replace_names=[], 
        p=0.1
        ):
        for name, module in list(self.module.named_modules()):
            parent, child_name = get_direct_parent(self.module, name)
            noisy_activations = False
            noisy_parameters = False
            actual_module = module
            if type(module) in activation_types or name in activation_names:
                noisy_activations = True
            if name in replace_names or type(module) in replace_types:
                noisy_activations = True
                actual_module = torch.nn.Identity()
            if type(module) in parameter_types or name in parameter_names:
                noisy_parameters = True
            if noisy_activations or noisy_parameters:
                setattr(parent, child_name, BernoulliLayer(actual_module, noisy_activations=noisy_activations,noisy_parameters=noisy_parameters, p=p))
    