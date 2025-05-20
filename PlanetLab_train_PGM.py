import torch
import torch.nn as nn
from torch.autograd import Variable, grad

from file_utils import *
from copy import deepcopy
from tqdm import tqdm
import sys
import scipy.io as sio
import numpy as np
import torch.nn.functional as F
class SelfAttention(nn.Module):
    """ A simple self-attention block. """
    def __init__(self, in_features):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)

    def forward(self, x):
        Q = self.query(x)  # Queries
        K = self.key(x)  # Keys
        V = self.value(x)  # Values
        # Compute the attention scores
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        # Apply the attention weights to the values
        weighted_values = torch.bmm(attention_weights, V)
        return weighted_values

class Generator(nn.Module):  # Attention
    def __init__(self, in_dim=70, out_dim=120 ** 2):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Input linear layer to transform the input vector
        self.input_layer = nn.Linear(in_dim, in_dim)
        # Attention block
        self.attention = SelfAttention(in_dim)
        # Output layer to flatten and map to desired output size
        self.output_layer = nn.Sequential(
            nn.Linear(in_dim * in_dim, out_dim),
            nn.Tanh()
        )

    def forward(self, input):
        # Transform input
        x = self.input_layer(input)
        x = x.unsqueeze(1).repeat(1, self.in_dim, 1)  # Repeat the input across a new dimension to form a square matrix
        # Apply self-attention
        x = self.attention(x)
        # Flatten the output from the attention layer
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, in_dim * in_dim)
        # Output transformation to match the required output size
        output = self.output_layer(x)
        return output

class Generator2(nn.Module):  # RNN
    def __init__(self, in_dim=70, out_dim=120**2, hidden_dim=512, num_layers=2):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # RNN layer
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # Output layer to produce the final output vector from the last hidden state
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh()  # Tanh to bound the output similar to your original Generator
        )

    def forward(self, input):
        # Reshape input to sequence format (batch_size, sequence_length, input_size)
        # Here sequence_length equals in_dim and input_size is 1
        x = input.view(-1, self.in_dim, 1)

        # RNN forward pass
        # Output will be (batch, seq_len, num_directions * hidden_size)
        # We use only the last hidden state
        _, h_n = self.rnn(x)

        # h_n shape is (num_layers, batch, hidden_size)
        # We take the last layer's hidden state
        last_hidden = h_n[-1]

        # Pass the last hidden state through the fully connected layer
        output = self.fc(last_hidden)

        return output

class Generator1(nn.Module):  # cnn
    def __init__(self, in_dim=70, out_dim=120 ** 2, mid_channels=64):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Input layer - Linear transformation
        self.fc_input = nn.Linear(in_dim, 24 * 24 * mid_channels)  # Adjust size to match

        # CNN Layers
        self.conv_blocks = nn.Sequential(
            # First Convolution Block
            nn.ConvTranspose2d(mid_channels, mid_channels // 2, kernel_size=4, stride=2, padding=1),
            # Output size: 48x48
            nn.BatchNorm2d(mid_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Second Convolution Block
            nn.ConvTranspose2d(mid_channels // 2, mid_channels // 4, kernel_size=4, stride=2, padding=1),
            # Output size: 96x96
            nn.BatchNorm2d(mid_channels // 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer to match desired output size
            nn.Conv2d(mid_channels // 4, 1, kernel_size=3, stride=1, padding=1),  # Output size remains 96x96
            nn.Tanh()
        )

        # Final layer to adjust to the required output dimensions (flat vector)
        self.fc_output = nn.Linear(96 * 96, out_dim)  # Adjust size to match output size

    def forward(self, input):
        x = self.fc_input(input)
        x = x.view(-1, 64, 24, 24)  # Reshape into a suitable tensor for convolution layers
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)  # Flatten the output for final layer
        output = self.fc_output(x)
        return output

class Generator0(nn.Module):
    def __init__(self, in_dim=70, out_dim=120 ** 2, mid_dim=1024):
        super(Generator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mid_dim = mid_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, mid_dim), nn.LeakyReLU(),
            nn.Linear(self.mid_dim, self.mid_dim), nn.LeakyReLU(),
            nn.Linear(self.mid_dim, self.out_dim), nn.Tanh()
        )

    def forward(self, input):
        return self.net(input)


class Measurement(nn.Module):
    def __init__(self, in_dim=120 ** 2, out_dim=25, mid_dim=500, bs=2, device=torch.device("cuda:0")):
        super(Measurement, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.bs = bs
        self.out_dim = out_dim
        self.mid_dim = mid_dim

        # generate measurement matrix
        # missing_rate = 0.4
        eye = np.eye(in_dim).astype(np.float32)
        np.random.seed()
        sample = np.random.choice(in_dim, in_dim - out_dim, replace=False)
        measure1 = np.delete(eye, (sample), axis=0)
        # measure12=np.expand_dims(measure1,axis=0)
        # npmeasure=np.repeat(measure12,bs,axis=0)
        self.measure = torch.from_numpy(measure1).to(device)

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.mid_dim), nn.LeakyReLU(),
            nn.Linear(self.mid_dim, self.mid_dim), nn.LeakyReLU(),
            nn.Linear(self.mid_dim, self.out_dim)
        )

    def forward(self, input):
        a = input.unsqueeze(dim=-1)
        b = torch.mm(self.measure, a[0, :, :])
        b1 = b.unsqueeze(dim=0)
        for i in range(1, self.bs):
            c = torch.mm(self.measure, a[i, :, :])
            c1 = c.unsqueeze(dim=0)
            b1 = torch.cat([b1, c1], 0)

        d = b1.reshape(self.bs, -1)
        return d


class Policy(nn.Module):
    def __init__(self, in_dim=125, out_dim=100, mid_dim=512):
        super(Policy, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mid_dim = mid_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, mid_dim), nn.LeakyReLU(),
            nn.Linear(self.mid_dim, self.mid_dim), nn.LeakyReLU(),
            nn.Linear(self.mid_dim, self.out_dim), nn.Tanh()
        )

    def forward(self, input):
        x = self.net(input)
        x = x / x.norm(dim=-1, keepdim=True)
        return x


class Step_size(nn.Module):
    def __init__(self, initial_step_size=np.log(0.01)):
        # self.s = nn.Parameter(torch.tensor(0.01))
        # Other necessary setup
        super(Step_size, self).__init__()
        self.s = nn.Parameter(torch.tensor(initial_step_size))

    def forward(self, ):
        # Necessary forward computations
        return 1


def load_data(batch_size):
    dataFile = './PlanetLab_RTT.mat'
    valData = sio.loadmat(dataFile)
    data1 = valData['T']
    data2 = data1.transpose(2, 0, 1)
    np.random.seed(1)
    data = np.random.permutation(data2)
    data = data[:, 0:120, 0:120].astype(np.float32)
    data = data / data.max()
    y = data[0:14, :, :]
    # torch_x= torch.from_numpy(x.T)
    torch_y = torch.from_numpy(y)
    torch_dataset = torch.utils.data.TensorDataset(torch_y)

    train_loader = torch.utils.data.DataLoader(dataset=torch_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)
    return train_loader


def train_dcs(latent_dim=70, batch_size=2, num_training_epoch=61, lr=1e-4, initial_step_size=0.01, num_grad_iters=4,
              device=torch.device("cuda:0")):
    # num_training_epoch = 100000------->5
    file_exporter = FileExporter('./image', )
    training_data = load_data(batch_size)
    measurement_out_dim = int(120*120*0.9)
    gen = Generator().to(device)
    measurement = Measurement(out_dim=measurement_out_dim).to(device)
    step_size = Step_size().to(device)  # torch.nn.Parameter(torch.ones(1) * np.log(initial_step_size)).to(device)
    optimizer = torch.optim.Adam(list(gen.parameters()) + list(step_size.parameters()), lr=lr)
    policy = Policy(in_dim=(latent_dim + measurement_out_dim)).to(device)
    MSELoss = nn.MSELoss()
    pbar = tqdm(range(num_training_epoch))
    n_batch = 0
    z_0 = torch.randn(batch_size, latent_dim, device=device, requires_grad=False)

    for epoch in pbar:
        for i, data in enumerate(training_data):
            images = data[0]
            if (images.shape[0] == 32):
                continue
            original_data = images.reshape(batch_size, -1).to(device)
            # original_data = (original_data) * 2 - 1
            z_initial = torch.randn(batch_size, latent_dim, device=device, requires_grad=True)
            measurement_original_data = measurement(original_data)
            z = torch.clone(z_initial)
            z = z / z.norm(keepdim=True, dim=-1)

            for itr in range(1, num_grad_iters):
                t = measurement(gen(z))
                e = t - measurement_original_data
                l = torch.mul(e, e).sum(dim=-1)+ z.norm(dim=-1)
                g = grad(outputs=l, inputs=z, grad_outputs=torch.ones(2, device=device))[0]
                # z = z - step_size.s.exp() * g
                z = z - 0.01 * g
                z = z / z.norm(dim=-1, keepdim=True)
                # z = policy(torch.cat((z, measurement_original_data), dim=-1))

            z_optimized = z
            generated_data_initial = gen(z_initial)
            generated_data_optimized = gen(z_optimized)
            measurement_generated_data_initial = measurement(generated_data_initial)
            measurement_generated_data_optimized = measurement(generated_data_optimized)
            em = measurement_generated_data_optimized - measurement_original_data
            generated_loss = torch.mul(em, em).sum(dim=-1).mean()
            R1 = ((measurement_generated_data_initial - measurement_original_data).reshape(-1,
                                                                                           measurement_out_dim).norm(
                dim=-1) +
                  (generated_data_initial - original_data).norm(dim=-1))
            R2 = ((measurement_generated_data_optimized - measurement_original_data).reshape(-1,
                                                                                             measurement_out_dim).norm(
                dim=-1) +
                  (generated_data_optimized - original_data).norm(dim=-1))
            R3 = ((measurement_generated_data_optimized - measurement_generated_data_initial).reshape(-1,
                                                                                                      measurement_out_dim).norm(
                dim=-1) +
                  (generated_data_optimized - generated_data_initial).norm(dim=-1))
            RIP_loss =torch.mul(R1, R1) + 0*torch.mul(R2, R2) + torch.mul(R3, R3)

            # l2_regularization = torch.tensor(0.0).to(device)
            # for param in gen.parameters():
            #     l2_regularization += torch.norm(param, 2)

            # loss = generated_loss+l2_regularization
            loss =  RIP_loss.mean()
            # loss = generated_loss

            if True:
                RECON_LOSS = (generated_data_optimized - original_data).norm(dim=-1).mean()
                torch.save(gen.state_dict(), "PLB_gen_dmc.pth")

                pr = (z_optimized - z_initial)
                desc = f"nbatch: {n_batch} | RIP_LOSS: {(RIP_loss.mean() / 3):.2f} | GEN_LOSS: {generated_loss:.2f} | RECON_LOSS: {RECON_LOSS:.2f} | step_size: {step_size.s.exp().item():.5f} | optim_cost: {torch.mul(pr, pr).sum(dim=-1).mean():.2f} | z_mean:{z_initial.norm(dim=-1).mean():.2f}|z_opt.mean: {z_optimized.norm(dim=-1).mean():.2f}"
                pbar.set_description(desc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_batch += 1


if __name__ == "__main__":
    # import wandb
    # config = {
    #     'method': 'cs',
    # 'backend': 'pytorch',
    # 'dataset': 'MNIST',
    #     'latent_dim': 100,
    # }
    # wandb.init(project="compressive_sensing", entity="graph-tensor", sync_tensorboard=True, config=config, name='compressive_sensing', monitor_gym=True, save_code=True)

    train_dcs()
