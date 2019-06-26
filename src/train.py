#!/usr/bin/env python
from os.path import join
from torch import optim
from torch.utils.data import DataLoader

from src.baxter_config import JOINT_NAMES, get_headers_with_collision,get_joint_names, get_limb_headers, get_joint_limits
from src.vae import VAE, train, test, generate_samples
from src.utils import normalize_data, plot_loss, write_samples_to_csv, find_data_file, get_path, parse_args, tic, toc
from src.path_config import input_base_path, output_base_path, model_base_path, result_base_path

import argparse
import pandas as pd
import math
import numpy as np
import os
import torch

def load_and_preprocess_data(file_name, headers, joint_limits, batch_size, normalization=False):
    """
    Load and preprocess data; 
    """
    print("Loading data from file: %s" % file_name)
    # Set headers;
    input_data = pd.read_csv(file_name, usecols=headers)

    # Additional step to preserve the header order when reading data;
    data = input_data[headers].values

    print("The first 5 values from the dataset are: %s" % data[0:5, :])
    X = data[:, :-1]
    y = data[:, -1]

    # normalize data if needed
    if normalization:
        X = normalize_data(X, joint_limits)
        print("The first 5 values after normalization are: %s" % X[0:5, :])

    print("The max of X is: %s; The min of X is: %s" % (np.max(X), np.min(X),))

    cutoff = math.ceil(X.shape[0] * 0.8) # 80 percent training data
    train_X = X[:cutoff, :]
    test_X = X[cutoff:, :]
    train_y = y[:cutoff]
    test_y = y[cutoff:]

    # train_free_y = train_y[train_y == 1]
    train_free_X = train_X[train_y == 1]
    print("The number of samples in train_free is: %s" % train_free_X.shape[0])

    test_free_X = test_X[test_y == 1]
    # test_free_y = test_y[test_y == 1]
    print("The number of samples in test_free is: %s" % test_free_X.shape[0])

    train_loader = DataLoader(train_free_X, batch_size=batch_size,
                            shuffle=True, num_workers=1)
    test_loader = DataLoader(test_free_X, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, test_loader

def main(args):
    # Hyperparamters Example
    """
    batch_size = 1000
    learning_rate = 0.001
    d_input = num_joints
    h_dim1 = 256
    h_dim2 = 100
    kld_weight = 0.5
    generated_sample_size = 1000000
    epochs = 10
    # Latent_variables
    d_output = 7
    """
    print("The arguments are: \n" + str(args))

    label = args.label
    env = args.env

    # hyperparams param;
    beta = args.beta
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_joints = args.num_joints
    h_dim1 = args.h_dim1
    h_dim2 = args.h_dim2
    d_output = args.d_output
    epochs = args.epochs
    generated_sample_size = args.generated_sample_size

    # Get headers and joint limits
    selected_joints = JOINT_NAMES[0:num_joints]
    headers_with_collision = get_headers_with_collision(selected_joints, env, label)
    print("The headers with collisions are: %s" % headers_with_collision)

    joint_limits = get_joint_limits(selected_joints)
    print("The joint limits are: %s" % joint_limits)

    # Load data
    data_file_name = find_data_file(input_base_path, num_joints)
    train_loader, test_loader = load_and_preprocess_data(data_file_name, headers_with_collision, joint_limits,
                                                         batch_size, normalization=False)

    # Process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(num_joints, h_dim1, h_dim2, d_output).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_losses = []
    test_loss = math.inf
    for epoch in range(1, epochs + 1):
      start = tic()
      mu, logvar, train_loss, recon_loss, kld = train(model=model, optimizer=optimizer, device=device, epoch=epoch,
                                                      train_loader=train_loader, kld_weight = beta)
      toc(start)
      training_losses.append((train_loss, recon_loss, kld))
      test_loss = test(model=model, epoch=epoch, device=device, test_loader=test_loader, kld_weight = beta)
      toc(start)

    # Print out metric for evaluation.
    print("Final average test loss: {:.4f};".format(test_loss))
    total_loss, recon_loss, kld_loss = zip(*training_losses)
    t = range(1, epochs + 1)

    path_args = parse_args(args)
    result_dir = get_path(result_base_path, path_args)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plot_loss(t, total_loss, recon_loss, kld_loss, result_dir)

    # Sample from data
    configs_written = 0
    write_header = True
    joint_names = get_joint_names(selected_joints)
    complete_headers = get_limb_headers('right')

    # Find the path to write to;
    output_dir = get_path(output_base_path, path_args)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = join(output_dir, "right_" + str(num_joints) + '_' + str(generated_sample_size) + '.csv')
    print("Writing samples to: %s" % (file_name))

    size = batch_size
    # write data to file;
    for i in range(math.ceil(generated_sample_size / batch_size)):
        print("Writing the %dth batch" % i)
        if generated_sample_size - configs_written < batch_size:
            size = generated_sample_size - configs_written
        samples = generate_samples(size, d_output, device, model)
        if i > 0:
            write_header = False
        write_samples_to_csv(file_name, samples, joint_names,
                             complete_headers, write_header=write_header, print_frame=False)
        configs_written += size

    # Write model artifacts to directory;
    model_directory = get_path(model_base_path, path_args)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    with open(os.path.join(model_directory, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

    return file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset configurations
    parser.add_argument('--label', default='self')
    parser.add_argument('--env')

    # hyperparameters
    parser.add_argument('--h-dim1', type=int, default=256)
    parser.add_argument('--h-dim2', type=int, default=100)
    parser.add_argument('--d-output', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--use-cuda', type=bool, default=False)
    parser.add_argument('--beta', type=float, default=1.0)

    # Fixed static parameters;
    parser.add_argument('--num-joints', type=int, default=7)
    parser.add_argument('--generated-sample-size', type=int, default=1000000)

    args, _ = parser.parse_known_args()
    main(args)


