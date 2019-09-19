from torch.nn import functional as F
from torch import nn

import torch

class VAE(nn.Module):

    def __init__(self, d_in, h_dim1, h_dim2, d_out, keep_prob=0):
        super(VAE, self).__init__()
        # The latent variable is defined as z = N(\mu, \sigma).
        self.d_in = d_in

        # Encoder network
        self.fc1 = torch.nn.Linear(d_in, h_dim1)
        self.fc2 = torch.nn.Linear(h_dim1, h_dim2)
        self.mean = torch.nn.Linear(h_dim2, d_out)
        self.logvar = torch.nn.Linear(h_dim2, d_out)

        # Decoder network
        self.fc3 = torch.nn.Linear(d_out, h_dim2)
        self.fc4 = torch.nn.Linear(h_dim2, h_dim1)
        self.fc5 = torch.nn.Linear(h_dim1, d_in)

    # Encode the input into a normal distribution
    def encode(self, x):
        h1 = F.elu(self.fc1(x))
        h2 = F.elu(self.fc2(h1))
        return self.mean(h2), self.logvar(h2)

    # Reparameterize the normal;
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h2 = F.elu(self.fc3(z))
        h1 = F.elu(self.fc4(h2))
        # No need to normalize when using mse loss;
        # return torch.sigmoid(self.fc5(h1))
        return self.fc5(h1)

    # Forward pass;
    def forward(self, x):
        # Change the array to float first;
        mu, logvar = self.encode(x.view(-1, self.d_in))
        # print("The value of mu and logvars are: %s, %s" % (mu, logvar))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function_bce(recon_x, x, mu, logvar, beta, num_joints):
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, num_joints), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * KLD, recon_loss, KLD

# Reconstruction + KL divergence losses summed over all elements and batch
# Beta is a parameter for balancing between kld and recon cost;
# See paper: https://pdfs.semanticscholar.org/a902/26c41b79f8b06007609f39f82757073641e2.pdf
def loss_function_mse(recon_x, x, mu, logvar, beta, num_joints):
    mse = nn.MSELoss(reduction='sum')
    recon_loss = mse(recon_x, x.view(-1, num_joints))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * KLD, recon_loss, KLD


def train(model, optimizer, device, epoch, train_loader, kld_weight):
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    for batch_idx, configs in enumerate(train_loader):
        configs = configs.to(device).float()
        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, logvar = model(configs)
        # print("Max in the reconstructed_batch: %s; Max in the original dataset: %s" %  (np.max(recon_batch.detach().numpy()), np.max(configs.detach().numpy()),))
        # print("Min in the reconstructed_batch: %s; Min in the original dataset: %s" % (np.min(recon_batch.detach().numpy()), np.min(configs.detach().numpy()),))

        # Customize different cost function here.
        # loss, bce, kld = loss_function_bce(recon_batch, configs, mu, logvar)
        loss, rec, kld = loss_function_mse(recon_batch, configs, mu, logvar, kld_weight, model.d_in)

        train_loss += loss.item()
        recon_loss += rec.item()
        kld_loss += kld.item()

        # Backward pass
        loss.backward()
        optimizer.step()
    print('====> Epoch: {} Average total training_Loss: {:.4f}; Reconstruction loss: {:.4f}; KLD: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset),
               recon_loss / len(train_loader.dataset),
               kld_loss / len(train_loader.dataset)))
    return mu, logvar, train_loss / len(train_loader.dataset), recon_loss / len(train_loader.dataset), kld_loss / len(
        train_loader.dataset)


def test(model, epoch, device, test_loader, kld_weight):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, configs in enumerate(test_loader):
            configs = configs.to(device).float()
            recon_batch, mu, logvar = model(configs)
            # Customize different cost functions here.
            # loss, bce, kld = loss_function_bce(recon_batch, configs, mu, logvar)
            loss, recon_loss, kld = loss_function_mse(recon_batch, configs, mu, logvar, kld_weight, model.d_in)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Epoch: {} Test_loss: {:.4f};'.format(epoch, test_loss))
    return test_loss


def generate_samples(generated_batch_size, d_output, device, model):
    # Generate sampled outputs;
    with torch.no_grad():
        norms = torch.randn(generated_batch_size, d_output).to(device)
        samples = model.decode(norms).cpu().numpy()
    return samples