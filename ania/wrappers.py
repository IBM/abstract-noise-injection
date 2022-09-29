import torch
from ania.base_regularizer import BaseRegularizer
from ania.bernoulli import BernoulliReg
from ania.noise_injection import NoiseInjection
from ania.utils import get_direct_parent
from criterion import make_criteria

def noise_injection(
        model, 
        noisy_parameter_types, 
        noisy_parameter_names, 
        noisy_activation_types, 
        noisy_activation_names, 
        existing_dropout_action,
        parameter_epsilon_type,
        parameter_epsilon_args,
        activation_epsilon_type,
        activation_epsilon_args,
        outer_beta,
        outer_lex,
        base_criteria_config,
        standard_criterion,
        n_samples,
        sample_fuzziness,
        sample_corner_deflation,
        init_sample_scale,
        post_step_sample_scale,
        inner_beta,
        inner_lex,
        step_p,
        step_size,
        n_steps
    ):
    inner_losses = None
    if not inner_beta is None:
        inner_losses = make_criteria(
                beta = inner_beta,
                lex = inner_lex,
                base_config=base_criteria_config,
                standard_criterion=standard_criterion()
        )
    
    outer_losses = make_criteria(
            beta = outer_beta,
            lex = outer_lex,
            base_config=base_criteria_config,
            standard_criterion=standard_criterion()
        )

    reg_model = NoiseInjection(
        model,
        outer_losses,
        n_samples,
        sample_fuzziness,
        sample_corner_deflation,
        init_sample_scale,
        post_step_sample_scale,
        inner_losses,
        step_p,
        step_size,
        n_steps
        )
    replace_types = []
    if existing_dropout_action == "replace":
        replace_types.append(torch.nn.Dropout)
    if existing_dropout_action == "delete":
        for name, module in reg_model.named_modules():
            if type(module) is torch.nn.Dropout:
                parent, child_name = get_direct_parent(reg_model, name)
                setattr(parent, child_name, torch.nn.Identity)
    save_act = False
    if "act_entr" in outer_losses or  (not inner_losses is None and "act_entr" in inner_losses):
        save_act = True
    if "act_prior" in outer_losses or (not inner_losses is None and "act_prior" in inner_losses):
        save_act = True
    reg_model.set_noisy(
        activation_types=noisy_activation_types,
        activation_names=noisy_activation_names,
        parameter_types = noisy_parameter_types,
        parameter_names=noisy_parameter_names,
        replace_types=replace_types,
        save_act = save_act,
    )

    if not parameter_epsilon_type is None:
        reg_model.set_parameter_epsilon(parameter_epsilon_type, **parameter_epsilon_args)
    
    if not activation_epsilon_type is None:
        reg_model.set_activation_epsilon(activation_epsilon_type, **activation_epsilon_args)

    return reg_model

def vanilla(model, standard_criterion):
    return BaseRegularizer(model, outer_losses=make_criteria(beta = {"standard":1.0}, lex = {"standard":0}, standard_criterion=standard_criterion(), base_config = {}), n_samples=1)

def bernoulli(
        model, 
        noisy_parameter_types, 
        noisy_parameter_names, 
        noisy_activation_types, 
        noisy_activation_names, 
        existing_dropout_action,
        outer_beta,
        outer_lex,
        base_criteria_config,
        standard_criterion,
        n_samples,
        dropout_p
):
    outer_losses = make_criteria(
            beta = outer_beta,
            lex = outer_lex,
            base_config=base_criteria_config,
            standard_criterion=standard_criterion()
        )

    reg_model = BernoulliReg(
        model,
        outer_losses,
        n_samples,
    )
    replace_types = []
    if existing_dropout_action == "replace":
        replace_types.append(torch.nn.Dropout)
    if existing_dropout_action == "delete":
        for name, module in reg_model.named_modules():
            if type(module) is torch.nn.Dropout:
                parent, child_name = get_direct_parent(reg_model, name)
                setattr(parent, child_name, torch.nn.Identity())
    reg_model.set_noisy(
        activation_types=noisy_activation_types,
        activation_names=noisy_activation_names,
        parameter_types = noisy_parameter_types,
        parameter_names=noisy_parameter_names,
        replace_types=replace_types,
        p=dropout_p
    )

    return reg_model