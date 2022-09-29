import torch
from ania.utils import lexicographical_backward

class BaseRegularizer(torch.nn.Module):

    def __init__(self, module, outer_losses, n_samples):
        super().__init__()
        self.module = module
        self.outer_losses = outer_losses
        self.n_samples = n_samples

    def reset_sample_shape(self, *args, **kwargs):
        pass

    def outer_params(self):
        for _, param in self.named_outer_params():
            yield param
    
    def named_outer_params(self):
        for name, param in self.named_parameters():
            yield name, param

    def inner(self, *args, y, y_0, **kwargs):
        return 0
    
    def inner_requires_grad(self):
        return False
    
    def ignore_noise(self, ignore):
        return
    
    def forward(self, *args, y=None, compute_y_star=True,**kwargs):
        y_0 = self._forward(*args, y_0=True, y_star=False, **kwargs)
        if not compute_y_star:
            return y_0
        self.inner(*args, y=y, y_0=y_0, **kwargs)
        return y_0, self._forward(*args, y_0=False, y_star=True, **kwargs)

    def _forward(self, *args, y_0=True, y_star=True,**kwargs):
        y_0_ = y_star_ = None
        if y_0:
            self.ignore_noise(True)
            y_0_ = self.module(*args, **kwargs)
        if y_star:
            self.ignore_noise(False)
            y_star_ = self.module(*args, **kwargs)
        if not y_0:
            return y_star_
        if not y_star:
            return y_0_
        return (y_0_, y_star_)

    def apply_backward(
        self,
        x,
        y,
        ):
        steps = 0
        outer_grads = {}
        for name, param in self.named_outer_params():
            outer_grads[name] = torch.zeros_like(param)
        for _ in range(self.n_samples):
            y_0 = self._forward(x, y_star=False)
            
            steps += self.inner(x,y=y,y_0=y_0)
            
            y_star = self._forward(x, y_0 = False)
            
            outers = {k:outer_loss(self,y,y_0,y_star) for k, outer_loss in self.outer_losses.items()}
            new_grads, outer_steps = lexicographical_backward(dict(self.named_outer_params()),outers.values())
            steps += outer_steps
            for key, val in new_grads.items():
                if val is None:
                    val = 0.0
                outer_grads[key] += val / self.n_samples
        
        for name, param in self.named_outer_params():
            param.grad = outer_grads[name]
        
        return y_0, y_star, outers, steps

    # def apply_backward(
    #     self,
    #     x,
    #     y,
    #     ):
    #     with MeasureTime("entire_pass"):
    #         steps = 0
    #         outer_grads = {}
    #         for name, param in self.named_outer_params():
    #             outer_grads[name] = torch.zeros_like(param)
    #         for _ in range(self.n_samples):
    #             with MeasureTime("calc_y_0"):
    #                 y_0 = self._forward(x, y_star=False)
                
    #             with MeasureTime("run_inner"):
    #                 steps += self.inner(x,y=y,y_0=y_0)
                
    #             with MeasureTime("calc_y_star"):
    #                 y_star = self._forward(x, y_0 = False)
                
    #             with MeasureTime("calc_loss"):
    #                 outers = {k:outer_loss(self,y,y_0,y_star) for k, outer_loss in self.outer_losses.items()}
    #             with MeasureTime("backwards"):
    #                 new_grads, outer_steps = lexicographical_backward(dict(self.named_outer_params()),outers.values())
    #             with MeasureTime("set_grads"):
    #                 steps += outer_steps
    #                 for key, val in new_grads.items():
    #                     if val is None:
    #                         val = 0.0
    #                     outer_grads[key] += val / self.n_samples
            
    #         for name, param in self.named_outer_params():
    #             param.grad = outer_grads[name]
            
    #         return y_0, y_star, outers, steps

# class MeasureTime:
#     def __init__(self, name):
#         self.name = name
    
#     def __enter__(self):
#         self.start = time.time()
    
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         elapsed = time.time() - self.start
#         Logger.append(f"time_{self.name}", elapsed)
#         print(f"TIME MEASUREMENT: {self.name} \t {elapsed:.5f} seconds")