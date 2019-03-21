from torch import optim
from torchvision.utils import save_image
from torchvision import transforms
from torch.nn import functional as F
from torch import nn
from PIL import Image

import matplotlib.pyplot as plt
import math, json,os,sys,traceback
import torch
import torch.utils.data as data

# prefix = '/opt/ml/'
prefix = '/Users/jamesdi/Dropbox/UCSD/Research/ARCLab/Code/VAE/mnist-vae-container/ml'
input_path  = os.path.join(prefix, 'input/data/')
output_path = os.path.join(prefix, 'output/')
model_path  = os.path.join(prefix, 'model/')
param_path  = os.path.join(prefix, 'input/config/hyperparameters.json')
data_path = os.path.join(prefix, 'input/config/inputdataconfig.json')

# Adapted from https://github.com/juliensimon/dlnotebooks/blob/master/pytorch/01-custom-container/mnist_cnn.py
class MNISTLoader(data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, path=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        # Loading local MNIST files in PyTorch format: training.pt and test.pt.
        if self.train:
            self.train_data, self.train_labels = torch.load(os.path.join(path, 'training/training.pt'))
        else:
            self.test_data, self.test_labels = torch.load(os.path.join(path, 'validation/test.pt'))

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class VAE(nn.Module):
  # Simple 2-layer network to tranform a 2-dimensional vector into a scalar;
  def __init__(self, d_in, h_dim1, h_dim2, d_out, keep_prob=0):
    super(VAE, self).__init__()
    # The latent variable is defined as z = N(\mu, \sigma).

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
    h1 = F.relu(self.fc1(x))
    h2 = F.relu(self.fc2(h1))
    return self.mean(h2), self.logvar(h2)

  # Reparametrize the normal;
  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    # print("z is: %s" %  eps.mul(std).add_(mu))
    # return eps
    return eps.mul(std).add_(mu)

  def decode(self, z):
    h2 = F.relu(self.fc3(z))
    h1 = F.relu(self.fc4(h2))
    return torch.sigmoid(self.fc5(h1))

  # Forward pass;
  def forward(self, x):
    # Change the array to float first;
    mu, logvar = self.encode(x.view(-1, 28 * 28))
    # print("The value of mu and logvars are: %s, %s" % (mu, logvar))
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar

def normalize_tensor(x):
    min_x = torch.min(x)
    range_x = torch.max(x) - min_x
    if range_x > 0:
      normalized = (x - min_x) / range_x
    else:
      normalized = torch.zeros(range_x.size())
    return normalized

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, optimizer, device, epoch, train_loader):
  model.train()
  train_loss = 0
  for batch_idx, (images, labels) in enumerate(train_loader):
    images = images.to(device).float()
    optimizer.zero_grad()

    # Forward pass
    recon_batch, mu, logvar = model(images)
    loss = loss_function(recon_batch, images, mu, logvar)
    # Backward pass
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
  print('====> Epoch: {} Average Training_Loss: {:.4f};'.format(
    epoch, train_loss / len(train_loader.dataset)))
  return mu, logvar

def test(model, epoch, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
      for i, (images, labels) in enumerate(test_loader):
        images = images.to(device).float()
        recon_batch, mu, logvar = model(images)
        test_loss += loss_function(recon_batch, images, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Epoch: {} Test_loss: {:.4f};'.format(epoch, test_loss))
    return test_loss

def reconstruct_images(num_images, data_loader, device, model, path):
  images, labels = iter(data_loader).next()
  images = images.to(device).float()
  # Forward pass
  mean, logvar = model.encode(images.view(-1, 28 * 28))
  z = model.reparameterize(mean, logvar)
  reconstructed_samples = model.decode(z).cpu()
  reconstructed_samples = reconstructed_samples.view(-1, 28, 28).detach().numpy()

  figure = plt.figure()
  sizex = round(math.sqrt(num_images))
  for i in range(num_images):
    plt.subplot(sizex, math.ceil(num_images / sizex), i + 1)
    plt.tight_layout()
    plt.imshow(reconstructed_samples[i], cmap='gray', interpolation='none')
    plt.title("Ground truth: {}".format(labels[i]))
    plt.xticks([])
    plt.yticks([])
  plt.savefig(path + 'reconstructed_original_image' + '.png')

def generate_samples(generated_batch_size, d_output, device, model, path):
  # Generate sampled outputs;
  with torch.no_grad():
    norms = torch.randn(generated_batch_size, d_output).to(device)
    samples = model.decode(norms).cpu()

    # Save generated images to output dir;
    save_image(samples.view(generated_batch_size, 1, 28, 28),
               path + "reconstructed_samples" + '.png')

def main():
    """
    # parser = argparse.ArgumentParser()
    #
    # # hyperparameters sent by the client are passed as command-line arguments to the script.
    # parser.add_argument('--d-hidden', type=int, default=256)
    # parser.add_argument('--d-output', type=int, default=20)
    # parser.add_argument('--batch-size', type=int, default=600)
    # parser.add_argument('--learning-rate', type=float, default=0.01)
    #
    # # Fixed static parameters;
    # parser.add_argument('--d-input', type = int, default= 784)
    # parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--use-cuda', type=bool, default=False)
    # parser.add_argument('--generated_batch_size', type=int, default=64)
    #
    # # Data, model, and output directories
    # parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    # parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    # parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    # parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    #
    # args, _ = parser.parse_known_args()
    """

    with open(param_path, 'r') as params:
      hyperParams = json.load(params)
      print("Hyper parameters: " + str(hyperParams))

    lr = float(hyperParams.get('lr', '0.01'))
    batch_size = int(hyperParams.get('batch_size', '600'))
    epochs = int(hyperParams.get('epochs', '10'))
    d_input = int(hyperParams.get('d_input', '784'))
    h_dim1 = int(hyperParams.get('h_dim1', '512'))
    h_dim2 = int(hyperParams.get('h_dim2', '256'))
    d_output = int(hyperParams.get('d_output', '20'))
    num_images = int(hyperParams.get('num_reconstructed_images', '16'))
    binarized = bool(hyperParams.get('binarized', '0'))

    # Read input data config passed by SageMaker
    with open(data_path, 'r') as params:
      inputParams = json.load(params)
    print("Input parameters: " + str(inputParams))

    # Other settings
    generated_batch_size = 64
    # momentum = 0.8
    # seed = 1
    # log_interval = 10
    # torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Load training data;

    transform = None

    if binarized:
      transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x > 0,
        lambda x: x.float()
      ])
    else:
      transform = transforms.ToTensor()

    train_loader = torch.utils.data.DataLoader(
      MNISTLoader(train=True,
                  transform=transform,
                  path=input_path),
      batch_size=batch_size, shuffle=True, **kwargs)

    # # Load test data;
    test_loader = torch.utils.data.DataLoader(
        MNISTLoader(train=False,
                    transform=transform,
                    path=input_path),
        batch_size=batch_size, shuffle=True, **kwargs)

    model = VAE(d_input, h_dim1, h_dim2, d_output).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    output_sample_dir = model_path + '/' + str(h_dim1) + '-' + str(h_dim2) + '-' + str(d_output) + '-' + str(batch_size) + '-' + \
              str(lr)
    if not os.path.exists(output_sample_dir):
      os.makedirs(output_sample_dir)

    test_loss = math.inf
    for epoch in range(1, epochs + 1):
      mu, logvar = train(model=model, optimizer=optimizer, device=device, epoch=epoch, train_loader=train_loader)
      test_loss = test(model=model, epoch=epoch, device=device, test_loader=test_loader)

    # Print out metric for evaluation.
    print("Final average test loss: {:.4f};".format(test_loss))

    # Reconstruct the first 10 test images;
    reconstruction_image_loader = torch.utils.data.DataLoader(
        MNISTLoader(train=False,
                transform = transform,
                path= input_path),
        batch_size=batch_size, shuffle=False, **kwargs)
    reconstruct_images(num_images, reconstruction_image_loader, device, model, model_path)

    # Generate samples
    generate_samples(generated_batch_size, d_output, device, model, model_path)

    # Save model artifact;
    if not os.path.exists(model_path):
      os.makedirs(model_path)
    with open(os.path.join(model_path, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__ =='__main__':
  try:
    main()
    sys.exit(0)
  except Exception as e:
    # Write out an error file. This will be returned as the failureReason in the
    # DescribeTrainingJob result.
    trc = traceback.format_exc()
    failure_path = os.path.join(output_path, 'failure')
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    with open(os.path.join(output_path, 'failure'), 'w') as s:
      s.write('Exception during training: ' + str(e) + '\n' + trc)
    # Printing this causes the exception to be in the training job logs, as well.
    print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
    # A non-zero exit code causes the training job to be marked as Failed.
    sys.exit(255)
