#!/usr/bin/env python
from os.path import join
from torch import optim
from torch.utils.data import DataLoader

from src.baxter_config import JOINT_NAMES, get_joint_names, get_collision_header, get_limb_headers, get_joint_limits
from src.vae import VAE, train, test, generate_samples
from src.utils import normalize_data, plot_loss, write_samples_to_csv, find_data_file, tic, toc
from src.path_config import INPUT_BASE_PATH, OUTPUT_BASE_PATH, SAMPLED_OUTPUT_TEMPLATE, get_run_id

import argparse
import pandas as pd
import math
import numpy as np
import os
import torch

END_EFFECTOR_NAMES = ['x', 'y', 'z']

def load_and_preprocess_data(file_name, data_headers, label_header, joint_limits, batch_size, normalization=False):
    """
    Load and preprocess data; 
    """
    print("Loading data from file: %s" % file_name)
    # Set headers;
    headers = data_headers + [label_header]
    input_data = pd.read_csv(file_name, usecols=headers)

    print("The first 5 values from the dataset are: %s" % input_data.values[0:5, :])

    # Additional step to preserve the header order when reading data;
    X = input_data[data_headers].values
    y = input_data[label_header].values

    print("The first 5 values from X are: %s" % X[0:5, :])
    print("The first 5 values from y are: %s" % y[0:5])

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

    d_input = num_joints

    h_dim1 = args.h_dim1
    h_dim2 = args.h_dim2
    d_output = args.d_output
    epochs = args.epochs
    generated_sample_size = args.generated_sample_size
    use_cuda = args.use_cuda
    include_pos = args.include_pos

    # Get headers and joint limits
    selected_joints = JOINT_NAMES[0:num_joints]

    data_headers = get_joint_names(selected_joints)
    label_header = get_collision_header(env, label)
    # headers_with_collision = get_headers_with_collision(selected_joints, env, label)

    if include_pos:
        data_headers += END_EFFECTOR_NAMES # add x,y,z into headers
        d_input += len(END_EFFECTOR_NAMES)
    print("The data headers are: %s" % data_headers)

    joint_limits = get_joint_limits(selected_joints)
    print("The joint limits are: %s" % joint_limits)

    # Load data
    data_file_name = find_data_file(INPUT_BASE_PATH % env, num_joints)
    train_loader, test_loader = load_and_preprocess_data(data_file_name, data_headers, label_header, joint_limits,
                                                         batch_size, normalization=False)

    # Process
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print("Using device: %s" % device)
    model = VAE(d_input, h_dim1, h_dim2, d_output).to(device)
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

    args.d_input = d_input
    run_id = get_run_id(args)
    output_directory = OUTPUT_BASE_PATH % (env, run_id)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plot_loss(t, total_loss, recon_loss, kld_loss, output_directory)

    # Sample from data
    configs_written = 0
    write_header = True

    joint_names = get_joint_names(selected_joints)
    complete_headers = get_limb_headers('right')
    if include_pos:
        joint_names += END_EFFECTOR_NAMES
        complete_headers += END_EFFECTOR_NAMES

    file_name = SAMPLED_OUTPUT_TEMPLATE % (env, run_id)
    print("Writing samples to: %s" % (file_name))

    size = batch_size
    # write data to file;
    for i in range(math.ceil(generated_sample_size / batch_size)):
        # print("Writing the %dth batch" % i)
        if generated_sample_size - configs_written < batch_size:
            size = generated_sample_size - configs_written
        samples = generate_samples(size, d_output, device, model)
        if i > 0:
            write_header = False
        write_samples_to_csv(file_name, samples, joint_names,
                             complete_headers, write_header=write_header, print_frame=False)
        configs_written += size

    with open(os.path.join(output_directory, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
    return output_directory

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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--include-pos', action='store_true')

    # Fixed static parameters;
    parser.add_argument('--num-joints', type=int, default=7)
    parser.add_argument('--generated-sample-size', type=int, default=1000000)

    # Later to be modified in the script
    parser.add_argument('--d_input', type=int)

    args, _ = parser.parse_known_args()
    output_directory = main(args)
    print("Result saved to: %s" % output_directory)
