import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hyper_parameters as parameters
# import parameter_calculator as calculator
from torch.utils.data import DataLoader, Dataset

from denoising_diffusion_pytorch import GaussianDiffusion1D, Trainer1D, Dataset1D, Unet1D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Unet1D(
    dim = 56,
    dim_mults = (1, 2, 4, 7),
    channels = 1
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 56,
    timesteps = 1000,
    objective = 'pred_v'
)

data = torch.load(str('results/model-1.pt'), map_location=device, weights_only=True)

diffusion.load_state_dict(data['model'])

novel_alloy = diffusion.sample(batch_size=1)

labelled_alloy = list(zip(parameters.element_list, np.around(novel_alloy.squeeze().tolist(),3).tolist()))

print(sorted(labelled_alloy, key=lambda x: x[1], reverse=True))