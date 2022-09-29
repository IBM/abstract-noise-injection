#%%
import torch
import numpy as np
from ania.base_regularizer import BaseRegularizer
from gen_norm import sample_gen_norm, log_pdf_gen_norm
from ania.utils import map_torch, StateLess, get_direct_parent, flatten_torch, lexicographical_backward
from ania.utils import device
#pdf_gen_norm(torch.Tensor([0]).reshape((1,1)), 0.5,0.5)

# %%



class AddNoise(torch.nn.Module):
    def __init__(self, get_eps=None):
        super().__init__()
        self.initialized = False
        self.get_eps = get_eps
        self.ignore = False
    
    def forward(self, inp):

        if self.ignore:
            return map_torch(lambda _: 0, inp)
        
        if not self.initialized:
            self.samp = map_torch(lambda mu: torch.zeros_like(mu, device=device), inp, param=True)
            # if issubclass(type(samp), torch.nn.Module):
            #     self.add_module("samp",samp)
            # else:
            #     self.register_parameter("samp",samp)
            self.initialized = True

        return map_torch(lambda eps, samp: eps*samp,self.get_eps(inp),self.samp, param=False)
        
        
class AddLayerNoise(torch.nn.Module):
    def __init__(self, module, get_eps=None, noisy_parameters=True, noisy_activations=True, save_act=False):
        super().__init__()
        self.noisy_parameters = noisy_parameters
        self.noisy_activations = noisy_activations
        sd = module.named_parameters() #named parameters?
        self.mu_params = torch.nn.ParameterDict(sd)
        if self.noisy_parameters:
            self.add_param_noise = AddNoise(get_eps)
        if self.noisy_activations:
            self.add_output_noise = AddNoise(get_eps)
        self.module = StateLess(module)
        self.save_act = save_act
    
    def forward(self, *args, **kwargs):
        w = self.get_noisy_param_vals()

        out = self.module(w, *args, **kwargs)
        if self.save_act:
            self.__dict__["mu_activations"] = out
        if self.noisy_activations:
            out= self.get_noisy_act_vals(out)
        return out
    
    def get_noisy_param_vals(self):
        if self.noisy_parameters:
            return map_torch(lambda x,y:x+y, self.add_param_noise(self.mu_params), self.mu_params)
        else:
            return self.mu_params
    
    def get_noisy_act_vals(self, mu=None):
        
        if mu==None:
            mu = self.mu_activations
        if not self.noisy_activations:
            return mu
        return map_torch(lambda x, y: x+y, mu, self.add_output_noise(mu))
    
    def get_param_noise(self):
        if self.noisy_parameters:
            return self.add_param_noise(self.mu_params)
        else:
            return self.mu_params
    
    def get_act_noise(self, mu = None):
        if mu==None:
            mu = self.mu_activations
        if not self.noisy_activations:
            return mu
        return self.add_output_noise(mu)
    
    # def __repr__(self):
    #     self.module.module.load_state_dict(dict(self.mu))
    #     out = super().__repr__()
    #     self.module.delete_state()
    #     return out



class ConstantEpsilon(torch.nn.Module):
    def __init__(self, c = 0.1, batched=False):
        super().__init__()
        self.c = c
    def forward(self, mu):
        return map_torch(lambda v: self.c * torch.ones_like(v, device=device), mu)

class MultiplicativeEpsilon(torch.nn.Module):
    def __init__(self, c = 0.1, batched=False, nograd=False):
        super().__init__()
        self.c = c
        self.nograd = nograd
    def forward(self, mu):
        def detach(v):
            if self.nograd:
                return v.detach()
            return v
        return map_torch(lambda v: self.c * torch.abs(detach(v)), mu)

class RelativeEpsilon(torch.nn.Module):
    def __init__(self, c = 0.1, p=2, batched=False, nograd=False):
        super().__init__()
        self.c = c
        self.p = p
        self.nograd = nograd
    def forward(self, mu):
        def detach(v):
            if self.nograd:
                return v.detach()
            return v
        return map_torch(lambda v: self.c * torch.norm(detach(v).reshape((-1,)),p=self.p) * torch.ones_like(v, device=device), mu)

class ParametricEpsilon(torch.nn.Module):
    def __init__(self, init_val=-10, batched=False):
        super().__init__()
        self.init_val = init_val
        self.initialized=False
        self.batched=batched

    def forward(self, mu):
        if not self.initialized:
            if self.batched:
                self.rho = map_torch(lambda v: self.init_val*torch.ones(v.shape[1:], device=device), mu, param=True)
            else:
                self.rho = map_torch(lambda v: self.init_val*torch.ones_like(v, device=device), mu, param=True)
            self.initialized=True
        return map_torch(lambda v: torch.log(torch.exp(v) + 1.0), self.rho)



class NoiseInjection(BaseRegularizer):
    def __init__(
            self,
            module,
            outer_losses,
            n_samples,
            sample_fuzziness,
            sample_corner_deflation,
            init_sample_scale,
            post_step_sample_scale,
            inner_losses,
            step_p,
            step_size,
            n_steps,
            save_mu_act = False,
            save_noisy_act = False
         ):
        super().__init__(module,outer_losses,n_samples)
        self.module = module
        self.sample_fuzziness = sample_fuzziness
        self.sample_corner_deflation = sample_corner_deflation
        self.init_sample_scale = init_sample_scale
        self.post_step_sample_scale = post_step_sample_scale
        self.inner_losses = inner_losses
        self.step_p = step_p
        self.step_size = step_size
        self.n_steps = n_steps

        self.save_mu_act = save_mu_act
        self.save_noisy_act = save_noisy_act
    
    def reset_sample_shape(self, *args, **kwargs):
        for module in self.module.modules():
            if type(module) is AddNoise:
                    module.initialized = False
        with torch.no_grad():
            self._forward(*args, **kwargs, y_0 = False)


    def set_parameter_epsilon(self, epsilon_module_class, *args, **kwargs):
        for module in self.module.modules():
            if type(module) is AddLayerNoise and hasattr(module, "add_param_noise"):
                module.add_param_noise.get_eps = epsilon_module_class(*args, **kwargs, batched=False)
    
    def set_activation_epsilon(self, epsilon_module_class, *args, **kwargs):
        for module in self.module.modules():
            if type(module) is AddLayerNoise and hasattr(module, "add_output_noise"):
                module.add_output_noise.get_eps = epsilon_module_class(*args, **kwargs, batched=True)
    
    def set_noisy(self, activation_types=[], activation_names=[], parameter_types=[], parameter_names=[], replace_types=[], replace_names=[], **kwargs):
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
                setattr(parent, child_name, AddLayerNoise(actual_module, noisy_activations=noisy_activations,noisy_parameters=noisy_parameters, **kwargs))
    
    def inner_params(self):
        for _, param in self.named_inner_params():
            yield param
    
    def named_inner_params(self):
        for name, module in self.named_modules():
            if type(module) is AddNoise:
                if issubclass(type(module.samp), torch.nn.Parameter):
                    yield f"{name}.samp", module.samp
                    continue
                for param_name, param in module.samp.named_parameters():
                    yield f"{name}.{param_name}", param
    
    def outer_params(self):
        for _, param in self.named_outer_params():
            yield param
    
    def named_outer_params(self):
        inner_params = list(self.inner_params())
        for name, param in self.named_parameters():
            if not any([param is p for p in inner_params]):
                yield name, param
    
    def ignore_noise(self, ignore):
        for module in self.module.modules():
            if type(module) is AddNoise:
                module.ignore = ignore

    def add_sample(self, fuzziness, corner_deflation, scale=1.0):
        for module in self.module.modules():
            if type(module) is AddNoise:
                with torch.no_grad():
                    module.samp = map_torch(lambda samp: samp + scale*(sample_gen_norm(samp.shape, fuzziness, corner_deflation).to(device)), module.samp, param=True)
    
    def reset_noise(self):
        for module in self.module.modules():
            if type(module) is AddNoise:
                with torch.no_grad():
                    module.samp = map_torch(lambda samp: torch.zeros_like(samp, device=device), module.samp, param=True)

    
    def sample_log_likelihood(self, fuzziness, corner_deflation, include_params, include_activations):
        acc = 0.0
        for module in self.module.modules():
            if type(module) is AddLayerNoise:
                samp = {}
                if include_params:
                    samp["params"] = module.get_param_noise()
                if include_activations:
                    samp["activations"] = module.get_act_noise()
                
                liks=map_torch(lambda samp: log_pdf_gen_norm(samp.reshape((1,-1)),fuzziness, corner_deflation),samp,param=False)
                for _, lik in flatten_torch(liks):
                    acc+=lik
        return acc
    
    def get_noisy_params(self):
        for module in self.module.modules():
            if type(module) is AddLayerNoise:
                yield module.get_noisy_param_vals()
    def get_noisy_activations(self):
        for module in self.module.modules():
            if type(module) is AddLayerNoise:
                yield module.get_noisy_act_vals()
    
    def get_mu_params(self):
        for module in self.module.modules():
            if type(module) is AddLayerNoise:
                yield module.mu_params
    
    def get_mu_activations(self):
        for module in self.module.modules():
            if type(module) is AddLayerNoise:
                yield module.mu_activations

    def reduce_noisy(self, func, include_params, include_activations):
        v_dash = {}
        if include_params:
            v_dash["params"] = self.get_noisy_params()
        if include_activations:
            v_dash["activations"] = self.get_noisy_activations()
            
        acc = 0.0
        for _, vv_dash in flatten_torch(v_dash):
            acc += func(vv_dash)
        return acc
    
    def reduce_mu(self, func, include_params, include_activations):
        v_dash = {}
        if include_params:
            v_dash["params"] = list(self.get_mu_params())
        if include_activations:
            v_dash["activations"] = list(self.get_mu_activations())
            
        acc = 0.0
        for _, vv_dash in flatten_torch(v_dash):
            acc += func(vv_dash)
        return acc
    
    def inner(self,*args,y, y_0, **kwargs):
        self.reset_noise()
        self.add_sample(fuzziness=self.sample_fuzziness, corner_deflation=self.sample_corner_deflation, scale=self.init_sample_scale)
        steps = 0
        for __ in range(self.n_steps):
            # print(1,float(self.inner_losses["noisy"](self,y,y_0,self._forward(x, y_0=False))))
            y_star = self._forward(*args, y_0=False, **kwargs)
            inners = [inner_loss(self,y,y_0,y_star) for inner_loss in self.inner_losses.values()]
            inner_grads, new_steps = lexicographical_backward(dict(self.named_inner_params()),inners)
            steps += new_steps
            for key, param in self.named_inner_params():
                inner_grad = inner_grads[key]
                if self.step_p is None:
                    param.grad = inner_grad
                else:
                    unnormed = torch.sign(inner_grad) * torch.abs(inner_grad)**(1/(self.step_p-1))
                    param.grad = unnormed / (unnormed.norm(p = self.step_p)+1e-10) #/ n_steps
            inner_sgd = torch.optim.SGD(self.inner_params(), lr=self.step_size, momentum=0.0, maximize=True)
            inner_sgd.step()
            # print(2,float(self.inner_losses["noisy"](self,y,y_0,self._forward(x, y_0=False))))
            if self.post_step_sample_scale >0:
                self.add_sample(self.sample_fuzziness,self.sample_corner_deflation,scale=self.post_step_sample_scale)
        return steps

    # def inner(self,*args,y, y_0, **kwargs):
    #     with MeasureTime("inner_reset"):
    #         self.reset_noise()
    #     with MeasureTime("initial_sample"):
    #         self.add_sample(fuzziness=self.sample_fuzziness, corner_deflation=self.sample_corner_deflation, scale=self.init_sample_scale)
    #     steps = 0
    #     for __ in range(self.n_steps):
    #         # print(1,float(self.inner_losses["noisy"](self,y,y_0,self._forward(x, y_0=False))))
    #         y_star = self._forward(*args, y_0=False, **kwargs)
    #         inners = [inner_loss(self,y,y_0,y_star) for inner_loss in self.inner_losses.values()]
    #         inner_grads, new_steps = lexicographical_backward(dict(self.named_inner_params()),inners)
    #         steps += new_steps
    #         for key, param in self.named_inner_params():
    #             inner_grad = inner_grads[key]
    #             if self.step_p is None:
    #                 param.grad = inner_grad
    #             else:
    #                 unnormed = torch.sign(inner_grad) * torch.abs(inner_grad)**(1/(self.step_p-1))
    #                 param.grad = unnormed / (unnormed.norm()+1e-10) #/ n_steps
    #         inner_sgd = torch.optim.SGD(self.inner_params(), lr=self.step_size, momentum=0.0, maximize=True)
    #         inner_sgd.step()
    #         # print(2,float(self.inner_losses["noisy"](self,y,y_0,self._forward(x, y_0=False))))
    #         if self.post_step_sample_scale >0:
    #             self.add_sample(self.sample_fuzziness,self.sample_corner_deflation,scale=self.post_step_sample_scale)
    #     return steps

    def inner_requires_grad(self):
        return self.n_steps >0


# #-----------------------------------------------------
# x = torch.randn(1, 3, 32, 32)
# y = torch.empty(1, dtype=torch.long).random_(10)



# net = Regularized(resnet32())
# net.set_noisy_types([nn.Linear, nn.Conv2d],noisy_parameters=True, noisy_activations=True)
# net.define_epsilon(ParametricEpsilon)
# net(x) #initialize activations

# CE = nn.CrossEntropyLoss()
# fuzziness=0.5
# corner_deflation=0.5


# sgd = torch.optim.SGD(net.outer_params(), lr=0.00000001, momentum=0.9)

# inner_losses = make_losses(
#     beta = {
#         "boundary":1.0
#     },
#     lex = {
#         "boundary":0
#     },
#     standard_loss=nn.CrossEntropyLoss(),
#     bases = loss_bases,
# )

# outer_losses = make_losses(
#     beta = {
#         "standard":1.0,
#         "boundary":1.0,
#         "act_entr":1.0,
#         "act_prior":1.0,
#         "param_entr":1.0,
#         "param_prior":1.0,
#     },
#     lex = {
#         "standard":0,
#         "boundary":0,
#         "act_entr":0,
#         "act_prior":0,
#         "param_entr":0,
#         "param_prior":0,
#     },
#     standard_loss = nn.CrossEntropyLoss(),
#     bases = loss_bases,
#     base_config={
#         "act_entr":{
#             "fuzziness":0.5,
#             "corner_deflation":0.5
#         },
#         "act_prior":{
#             "fuzziness":0.5,
#             "corner_deflation":0.5,
#             "use_mu":False,
#             "scale":1.0,
#         },
#         "param_entr":{
#             "fuzziness":0.5,
#             "corner_deflation":0.5,
#         },
#         "param_prior":{
#             "fuzziness":0.5,
#             "corner_deflation":0.5,
#             "use_mu":False,
#             "scale":1.0,
#         }
#     }
# )

# for _ in range(7):
#     sgd.zero_grad()
#     net.apply_backward(
#         n_steps=1,
#         step_size=1.0,
#         step_p=None,
#         init_sample_scale=1.0,
#         post_step_sample_scale=0.0,
#         fuzziness=0.5,
#         corner_deflation=0.5,
#         inner_losses=inner_losses,
#         outer_losses=outer_losses,
#         n_samples=1.0
#         )
#     sgd.step()

