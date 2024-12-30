import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hyper_parameters as parameters
import parameter_calculator as calculator
from torch.utils.data import DataLoader, Dataset
import tqdm
from einops import rearrange, reduce
from random import random

from denoising_diffusion_pytorch import GaussianDiffusion1D, Trainer1D, Dataset1D, Unet1D

# from DiT import DiT

# The path to save the trained generator model.
GEN_PATH = 'saved_models/generator_net.pt'
# The element compositions of the 278 existing CCAs.
cca_compositions = np.genfromtxt('data/train_composition.csv', delimiter=',')
# The empirical parameters of the 278 existing CCAs.
cca_parameters = np.loadtxt('data/train_parameter.csv', delimiter=',')
param_mean = cca_parameters.mean(axis=0)
param_std = cca_parameters.std(axis=0)

def exists(x):
    return x is not None

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# —————————————————————————————————— Customize the training set ————————————————————————————————————————
class TrainingSet(Dataset):
    """
    A customized Dataset used to train the cardiGAN model. It includes the element compositions and empirical
    parameters of the 278 exisitng CCAs.
    """

    def __init__(self):
        # Load the element compositions of the 278 existing CCAs.
        compositions = np.loadtxt('data/train_composition.csv', delimiter=',')
        # compositions = np.concatenate(
        #     (compositions, np.ones((compositions.shape[0], 1)) - 1), axis=1)
        
        #turning off the parameters as they should be in the loss function...

        # Load the empirical parameters and normalize it into a Gaussian distribution.
        # cca_params = np.loadtxt('data/train_parameter.csv', delimiter=',')
        # cca_params = (cca_params - param_mean) / param_std

        # Build the training set by concatenating the composition and parameter datasets.
        # self.data = torch.from_numpy(
        #     np.concatenate((compositions, cca_params[:, parameters.GAN_param_selection]), axis=1)).float()
        self.data = torch.from_numpy(compositions)
        self.len = compositions.shape[0]  # The length of the training set.

    def __getitem__(self, index):
        return self.data[index].to(torch.float).unsqueeze(0)

    def __len__(self):
        return self.len


# ————————————————————————————————— Define the neural networks ————————————————————————————————————
class DiffusionTransformerModel(nn.Module):
    def __init__(self, input_dim=56, hidden_dim=256, n_heads=8, n_layers=6):
        super(DiffusionTransformerModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Transformer Encoder
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim, batch_first=True)
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        
        # Transformer Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_layers)
    
    def forward(self, x, t, alphas_bar_sqrt):
        # encoded_x = self.encoder(x)
        noisy_x = q_sample(x, t, alphas_bar_sqrt)
        # expanded_t = t.view(-1, 1).expand(-1, self.input_dim)
        # combined_xt = torch.cat((noisy_x, expanded_t), dim=1)
        decoded_xt = self.decoder(noisy_x, x)

        return decoded_xt


def q_sample(x0, t, alphas_bar_sqrt):
    """
    Sample from q(x_t| x_0)
    :param x0: Initial data
    :param t: Time step
    :param alphas_bar_sqrt: Precomputed square root of alpha bar
    :return: Sampled data
    """
    noise = torch.randn_like(x0)
    alphas_t = alphas_bar_sqrt[t].view(-1, 1)  # Ensure alphas_t has shape [batch_size, 1]
    return alphas_t * x0 + (1 - alphas_t) * noise

# Precompute alphas_bar_sqrt values
import numpy as np

# betas = np.linspace(0.0001, 0.02, 20)
# alphas = 1. - betas
# alphas_bar = np.cumprod(alphas)
# alphas_bar_sqrt = torch.tensor(np.sqrt(alphas_bar), dtype=torch.float32)

def loss_function(model, x0, t, alphas_bar_sqrt):

    predicted_x0 = model(x0.float(), t, alphas_bar_sqrt)
    
    # Mean squared error loss
    mse_loss = F.mse_loss(predicted_x0, x0.float())
    
    # Add conditional logic if needed
    # For example, impose a constraint that the sum of weights should be 1
    condition_penalty = torch.abs(torch.sum(x0, dim=1) - 1)
    total_loss = mse_loss + torch.mean(condition_penalty)
    
    return total_loss.float()

#Subclass the gaussian diffusion class to incorporate the PI loss function
class PI_GaussianDiffusion1D(GaussianDiffusion1D):
    '''
    GaussianDiffusion1D with modified loss to incorproate phyiscal constraints.
    '''
    def p_losses(self, x_start, t, noise = None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)

        model_out = model_out.squeeze()
        
        # penalty for not outputting a valid composition that adds to 1:
        condition_penalty = torch.abs(torch.sum(model_out, dim=1) - 1)

        # penalty for any elements greater than threshold:
        threshold = 0.4
        penalty = torch.relu(model_out - threshold)

        # novel_alloy_norm = model_out

        # target lower values for the mean and s.d. of enthalpy of mixing - perhaps this should instead minimize the difference between the input alloy
        
        # enthalpy = torch.abs((calculator.calculate_enthalpy(model_out).view(model_out.shape[0], -1) - calculator.mean_param_dict['Enthalpy(kJ/mol)']) / calculator.std_param_dict['Enthalpy(kJ/mol)']) / calculator.max_param_dict['Enthalpy(kJ/mol)']
        # std_enthalpy = torch.abs((calculator.calculate_std_enthalpy(model_out, enthalpy).view(model_out.shape[0], -1) - calculator.mean_param_dict['std_enthalpy(kJ/mol)']) / calculator.std_param_dict['std_enthalpy(kJ/mol)']) / calculator.max_param_dict['std_enthalpy(kJ/mol)']
        # delta = torch.abs((calculator.calculate_delta(model_out).view(model_out.shape[0], -1) - calculator.mean_param_dict['Delta(%)']) / calculator.std_param_dict['Delta(%)']) / calculator.max_param_dict['Delta(%)']
        # entropy = (calculator.calculate_entropy(model_out).view(model_out.shape[0], -1) - calculator.mean_param_dict['Entropy(J/K*mol)']) / calculator.std_param_dict['Entropy(J/K*mol)']

        # return the loss as the mean across the batch
        return loss.mean()# + 0.00001*(condition_penalty.mean() + penalty.mean() + enthalpy.mean() + std_enthalpy.mean() + delta.mean()) / entropy.mean()

# ————————————————————————————————— Set up the neural networks ————————————————————————————————————

model = Unet1D(
    dim = 56,
    dim_mults = (1, 2, 4, 7),
    channels = 1
)

diffusion = PI_GaussianDiffusion1D(
    model,
    seq_length = 56,
    timesteps = 100,
    objective = 'pred_v'
)

if __name__ == "__main__":
    # ————————————————————————————————— Load the training set ————————————————————————————————————————

    training_set = TrainingSet()
    loader = DataLoader(dataset=training_set, batch_size=parameters.size_batch, shuffle=True)

    trainer = Trainer1D(
        diffusion, 
        dataset = training_set,
        train_batch_size=parameters.size_batch,
        train_lr=1e-5,
        train_num_steps=1000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=True
    )

    trainer.train()

    sampled_seq = diffusion.sample(batch_size = 4)
    print(sampled_seq.shape) # (4, 32, 128)
    # ————————————————————————————————— Start GAN training ————————————————————————————————————————————

    # train_epoch(diffusion_model, loader, alphas_bar_sqrt)

    # torch.save(diffusion_model.state_dict(), save_path)
