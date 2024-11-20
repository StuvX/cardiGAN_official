import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hyper_parameters as parameters
# import parameter_calculator as calculator
from torch.utils.data import DataLoader, Dataset

# The path to save the trained generator model.
GEN_PATH = 'saved_models/generator_net.pt'
# The element compositions of the 278 existing CCAs.
cca_compositions = np.genfromtxt('data/train_composition.csv', delimiter=',')
# The empirical parameters of the 278 existing CCAs.
cca_parameters = np.loadtxt('data/train_parameter.csv', delimiter=',')
param_mean = cca_parameters.mean(axis=0)
param_std = cca_parameters.std(axis=0)


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
        return self.data[index]

    def __len__(self):
        return self.len


# ————————————————————————————————— Define the neural networks ————————————————————————————————————
# In this effort we are using a diffusion model based on flow based transformer network (???)
class DiffusionModel(nn.Module):
    def __init__(self, input_dim=56, hidden_dim=256, n_steps=20):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        
        # Encoder: Maps the input data to a latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder: Maps from noise and time step back to the original data
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        encoded_x = self.encoder(x)
        expanded_t = t.view(-1, 1).expand(-1, self.hidden_dim)  # Expand t to match the feature dimension
        combined_xt = torch.cat((encoded_x, expanded_t), dim=1)
        decoded_xt = self.decoder(combined_xt)
        
        return decoded_xt

def q_sample(x0, t, alphas_bar_sqrt):
    """
    Sample from q(x_t| x_0)
    :param x0: Initial data
    :param t: Time step
    :param alphas_bar_sqrt: Precomputed sqrt(alphas_bar) values
    :return: Noisy samples at time step t
    """
    noise = torch.randn_like(x0)
    mean = alphas_bar_sqrt[t].unsqueeze(1) * x0
    return mean + (1 - alphas_bar_sqrt[t].unsqueeze(1))**0.5 * noise

# Precompute alphas_bar_sqrt values
betas = np.linspace(0.0001, 0.02, 20)
alphas = 1. - betas
alphas_bar = np.cumprod(alphas)
alphas_bar_sqrt = torch.tensor(np.sqrt(alphas_bar), dtype=torch.float32)


# class Classifier(nn.Module):
#     """
#     The phase classifier neural network of the cardiGAN model. The network used in the GAN training is pre-trained on
#     the 12 empirical parameters and reported phases of the 278 existing CCAs.
#     The reported phases are divided into 3 classes: single solid-solution, mixed solid-solution, solid-solution with
#     secondary phases.
#     """

#     def __init__(self):
#         super(Classifier, self).__init__()

#         # Set the model to have two latent layers with LeakyReLU activation functions.
#         self.model = nn.Sequential(
#             nn.Linear(12, 12),
#             nn.LeakyReLU(),
#             nn.Linear(12, 12),
#             nn.LeakyReLU(),
#             nn.Linear(12, 3),
#         )

#     def forward(self, x):
#         predicted_phase = self.model(x)
#         return predicted_phase

def loss_function(model, x0, t, alphas_bar_sqrt):

    noisy_x_t = q_sample(x0, t, alphas_bar_sqrt)
    
    # Forward pass through the model
    predicted_x0 = model(noisy_x_t.float(), t.float())
    
    # Mean squared error loss
    mse_loss = F.mse_loss(predicted_x0, x0.float())
    
    # Add conditional logic if needed
    # For example, impose a constraint that the sum of weights should be 1
    condition_penalty = torch.abs(torch.sum(x0, dim=1) - 1)
    total_loss = mse_loss + torch.mean(condition_penalty)
    
    return total_loss.float()

# ————————————————————————————————— Set up the neural networks ————————————————————————————————————

diffusion_model = DiffusionModel()

save_path = 'saved_models/diffusion_net.pt'

optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=parameters.lr_generator)

n_steps = 20


# ————————————————————————————————— Set up train functions ————————————————————————————————————————
def train_epoch(model, data_loader, alphas_bar_sqrt, epochs=10):
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, x0_batch in enumerate(data_loader):
            optimizer.zero_grad()
            t_batch = torch.randint(0, n_steps, (x0_batch.size(0),))

            loss = loss_function(model, x0_batch, t_batch, alphas_bar_sqrt)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')


if __name__ == "__main__":
    # ————————————————————————————————— Load the training set ————————————————————————————————————————

    training_set = TrainingSet()
    loader = DataLoader(dataset=training_set, batch_size=parameters.size_batch, shuffle=True, )

    # ————————————————————————————————— Start GAN training ————————————————————————————————————————————

    train_epoch(diffusion_model, loader, alphas_bar_sqrt)

    torch.save(diffusion_model.state_dict(), save_path)

