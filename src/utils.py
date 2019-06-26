from pandas import Series
from os.path import join, isfile
from src.baxter_config import get_limb_headers, get_joint_names, JOINT_NAMES
from src.path_config import test_output_base_path, PATH_ARGS

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

# Training configs
limit = 10000000  # set limit on number of points when reading the configurations

def tic():
    return time.time()

def toc(start):
    time_elapsed = time.time() - start
    print("Elasped time: %s" % time_elapsed)

def get_path(base_path, path_args):
    # Get the path for the output dir
    path = base_path
    for (arg, value) in path_args:
        path = path + str(value) + "_" + arg + "/"
    return path


def parse_args(args):
    """
    Parse out the args according to the order in PATH_ARGS, if they exist
    :param args: A namespace object.
    :return: A dictionary containing the file args.
    """
    path_args = []
    for arg in PATH_ARGS:
        if arg in args.__dict__.keys():
            path_args.append((arg, args.__dict__[arg]))
    return path_args


def find_data_file(path, num_joints):
    # Find the file to read data from;
    file_name = ''
    files = [f for f in os.listdir(path) if isfile(join(path, f))]
    for f in files:
        f_name = f.replace('.csv', '')
        joints = int(f_name.split('_')[1])
        num_samples = int(f_name.split('_')[2])
        if joints == num_joints and num_samples < limit:
            file_name = join(path, f)
            break
        elif num_samples > limit:
            raise IOError("Number of samples is greater than limit. Num of samples: " + str(num_samples))
    return file_name

# Data normalization
def normalize_data(X, limits):
    """
    Normalize the revolute joint data from joints limits to [0, 1]
    X: Data before normalization n x d array
    limits: joint limits for the revolute joints, a 2 x d array
    """
    X = X - limits[0, :]
    x_range = limits[1, :] - limits[0, :]
    return X / x_range


def recover_data(normalized_X, limits):
    """
    Recover the normalized X based on joint limits.
    normalized_X: the normalized data points;
    limits: the joint limits
    """
    x_range = limits[1, :] - limits[0, :]
    X = normalized_X * x_range
    return X + limits[0, :]


def test_normalize_data():
    print("Testing normalizing data...")
    X = np.array([
        [-1.7016, -2.147, -3.0541],
        [1.7016, 1.047, 3.0541],
        [0.087478421724874522, 0.49571075835981837, -0.50648066618520682]
    ])
    limits = np.array([
        [-1.7016, -2.147, -3.0541],
        [1.7016, 1.047, 3.0541]
    ])
    print(normalize_data(X, limits))


def test_recover_data():
    print("Testing recovering data...")
    X = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0.5, 0.5, 0.5]
    ])
    limits = np.array([
        [-1.7016, -2.147, -3.0541],
        [1.7016, 1.047, 3.0541]
    ])
    print(recover_data(X, limits))


def write_samples_to_csv(file_name, samples, sample_headers, headers, write_header=True, print_frame=False):
    """
    Write the samples to a csv
    """
    # recovered_samples = recover_data(samples, limits )
    df = pd.DataFrame(data=samples, columns=sample_headers)

    # Fill the rest of the headers
    for header in headers:
        if header not in sample_headers:
            df[header] = Series(np.zeros(samples.shape[0]), index=df.index)

    if print_frame:
        print("The recovered data frame is: %s" % df)
    mode = 'w'
    if not write_header:
        mode = 'a'
    df.to_csv(file_name, header=write_header, index=False, quoting=csv.QUOTE_NONNUMERIC, mode=mode)


def test_write_samples_to_csv():
    print("Testing writing samples to csv...")
    num_joints = 7
    beta = 1
    complete_headers = get_limb_headers('right')
    selected_headers = JOINT_NAMES[0:num_joints]
    headers = get_joint_names(selected_headers)

    samples = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    ])

    path_args = {
        "num_joints": num_joints,
        "beta": beta
    }
    output_dir = get_path(test_output_base_path, path_args)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    file_name = join(output_dir, "right_" + str(num_joints) + '.csv')
    write_samples_to_csv(file_name, samples, headers, complete_headers, write_header=True, print_frame=True)

def plot_loss(epochs, total, reconstruction, divergence, path):
    fig = plt.subplot(1, 1, 1)
    plt.plot(epochs, total, color = 'red', label = 'Total loss')
    plt.plot(epochs, reconstruction, color = 'blue', label = 'Reconstruction loss')
    plt.plot(epochs, divergence, color = 'green', label = 'Divergence Loss')
    plt.legend()
    plt.title("Training cost functions")
    plt.savefig(path + "training_loss.png")

def test_get_path():
    print("Testing path...")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--num_joints', type=int, default=7)
    parser.add_argument('--beta', type=int, default=1)

    args, _ = parser.parse_known_args()

    # Parse out the arguments
    path_args = parse_args(args)
    assert path_args.get('num_joints') == 7, "Parsed num of joints do not match, Expected:{}, Actual:{}"\
        .format(7, path_args.get('num_joints'))
    assert path_args.get('beta') == 1, "Parsed beta do not match, Expected:{}, Actual:{}"\
        .format(1, path_args.get('beta'))

if __name__ == "__main__":
    # Tests
    test_normalize_data()
    test_recover_data()
    test_write_samples_to_csv()
    test_get_path()
