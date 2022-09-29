from collections import Counter
import torch
from ania.gen_norm import log_pdf_gen_norm


def standard_criterion(criterion,net,y,y_0,y_star):
    return criterion(y_0,y)

def noisy_criterion(criterion,net,y,y_0,y_star):
    return criterion(y_star,y)

def boundary_criterion(criterion,net,y,y_0,y_star):
    return criterion(y_star,y_0)

def diff_criterion(criterion,net,y,y_0,y_star):
    return torch.abs(standard_criterion(criterion,net,y,y_0,y_star) - noisy_criterion(criterion,net,y,y_0,y_star))

def activation_entropy(criterion, net, y, y_0, y_star,fuzziness,corner_deflation):
    return net.sample_log_likelihood(fuzziness, corner_deflation, False, True)

def param_entropy(criterion, net, y, y_0, y_star,fuzziness,corner_deflation):
    return net.sample_log_likelihood(fuzziness, corner_deflation, True, False)

def activation_prior(criterion, net, y, y_0, y_star,fuzziness,corner_deflation, use_mu, scale):
    if use_mu:
        return net.reduce_mu(lambda x:log_pdf_gen_norm(x/scale,fuzziness,corner_deflation), False, True)
    return - net.reduce_noisy(lambda x:log_pdf_gen_norm(x/scale,fuzziness,corner_deflation), False, True)

def param_prior(criterion, net, y, y_0, y_star,fuzziness,corner_deflation, use_mu, scale):
    if use_mu:
        return - net.reduce_mu(lambda x:log_pdf_gen_norm(x/scale,fuzziness,corner_deflation), True, False)
    return - net.reduce_noisy(lambda x:log_pdf_gen_norm(x/scale,fuzziness,corner_deflation), True, False)



base_criteria = {
    "standard":standard_criterion,
    "noisy":noisy_criterion,
    "boundary":boundary_criterion,
    "diff":diff_criterion,
    "act_entr":activation_entropy,
    "param_entr":param_entropy,
    "act_prior":activation_prior,
    "param_prior":param_prior,
    }

def make_criteria(
    beta,
    lex,
    standard_criterion=torch.nn.CrossEntropyLoss(),
    bases=base_criteria,
    base_config={}
):
    def criterion(l):
        return lambda net,y,y_0,y_star: sum([beta[k] * bases[k](standard_criterion,net,y,y_0,y_star,**base_config.get(k,{})) for k in l])


    sorted_keys = [[k for k, ii in lex.items() if i == ii] for i in sorted(list(Counter(lex.values())))]

    return {"+".join(l) : criterion(l) for l in sorted_keys}