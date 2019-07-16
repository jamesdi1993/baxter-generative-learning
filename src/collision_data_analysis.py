#!/usr/bin/env python
# coding: utf-8
from src.baxter_config import JOINT_LIMITS, JOINT_NAMES, get_headers_with_collision, get_collision_header
from src.utils import find_data_file, get_path, parse_args
from src.path_config import result_base_path, input_base_path, validated_output_base_path

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def estimate_density(limits, num_bins, data):
    """
    args:
        limits: 2d array [[x1_low, x1_hight], [x2_low, x2_high]]
        num_bins: 1d array [x1_bins, x2_bins]
        data: The data samples, n x 2 array
    """
    counters = np.zeros(tuple(num_bins))
    res = (limits[:, 1] - limits[:, 0]) / num_bins
    # print("Resolution is: %s" % (res,))
    # print("The data is: %s" % (data,))
    indices = np.floor((data - limits[:, 0]) / res).astype(int)
    # print("The indices are: %s" % (indices,))
    for i in range(indices.shape[0]):
        try:
            counters[tuple(indices[i, :])] += 1
        except IndexError:
            print("The index is: %s" % (indices[i, :]))
            print("The data is: %s" % (data[i, :]))
            raise IndexError
    # print("The counters is: %s" % (counters, ))
    return counters / np.sum(counters)

def kl_divergence(limits, num_bins, x1, x2):
    """
    Compute the KL-Divergence of two distributions, with the same bounds;
    """
    p1 = estimate_density(limits, num_bins, x1).astype(np.float64)
    p2 = estimate_density(limits, num_bins, x2).astype(np.float64)
    epsilon = 0.1 * np.min(p2[p2 > 0])
    print("epsilon is: %s" % epsilon)
    print("Minimum of p2 is: %s" % np.min(p2))
    r = np.log(p1 / (p2 + epsilon))
    r[p1 == 0] = 0
    return p1.flatten().dot(r.flatten())

"""
Various statistics defined on the dataset
"""

def countSpillover(headers, dataset, limits):
    """
    Check for number of spillovers that are beyong hardware limits;
    """
    counts = {}
    for joint in headers:
        joint_limit = limits[joint]
        joint_header = 'right_' + joint
        joint_count = np.count_nonzero(np.logical_or(dataset[joint_header] < joint_limit[0],
                                                     dataset[joint_header] > joint_limit[1]))
        counts[joint_header] = joint_count / dataset.shape[0]
    return counts

def countSpilledProportion(headers, dataset, limits):
    values = np.zeros(dataset.shape[0])
    for joint in headers:
        joint_limit = limits[joint]
        joint_header = 'right_' + joint
        values = np.logical_or(values, dataset[joint_header] < joint_limit[0])
        values = np.logical_or(values, dataset[joint_header] > joint_limit[1])

    rows = values.nonzero()[0]
    print("The joint limits are: %s" % limits)
    print("Examples of samples that are outside of hardware limits: ")
    print(dataset.iloc[rows].iloc[0:5, :])
    # assert np.sum(dataset.iloc[rows][COLLISION_KEY] == np.zeros(len(rows))) == len(rows), \
        # "Some out of range configurations are collision-free!"
    return np.count_nonzero(values) / dataset.shape[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-joints', type=int, default=7)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--env')
    parser.add_argument('--label')
    parser.add_argument('--dry-run', action='store_true')

    args, _ = parser.parse_known_args()

    print("The arguments are: %s" % (args,))
    num_joints = args.num_joints
    env = args.env
    dry_run = args.dry_run
    label = args.label

    # Get the collision label
    collision_label = get_collision_header(env, label)

    path_args = parse_args(args)
    # print("The path args is: %s" % (path_args,))
    validated_output_path = get_path(validated_output_base_path % env, path_args)
    file_name = find_data_file(validated_output_path, num_joints)
    original_data_file_name = find_data_file(input_base_path % env, num_joints)

    output_dir = get_path(result_base_path % env, path_args)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    print("The validated file is: %s" % file_name)
    print("The output directory is: %s" % output_dir)
    print("The original data file is: %s" % original_data_file_name)

    if not dry_run:
        selected_joints = JOINT_NAMES[0:num_joints]
        headers_w_collision = get_headers_with_collision(selected_joints, env, label)
        data = pd.read_csv(file_name, usecols=headers_w_collision)

        for joint in JOINT_NAMES:
            joint_name = 'right_' + joint
            joint_value = data[joint_name]
            print("Statistics for %s" % joint_name)
            print("The maximum is: %s;" % np.max(joint_value))
            print("The minimum is: %s;" % np.min(joint_value))
            print("The range is: %s;" % (np.max(joint_value) - np.min(joint_value)))

            print("The mean is: %s;" % np.mean(joint_value))
            print("The median is: %s;" % np.median(joint_value))
            print("The joint limits for the range is: %s" % JOINT_LIMITS[joint])
            print("-----------------")
        i = 1

        """
        Plots for only the sampled data
        """
        fig = plt.figure(figsize=(10,10))
        for joint_name in JOINT_NAMES:
            plt.subplot(4, 2, i)
            header = 'right_' + joint_name
            joint_value = data[header]
            i += 1
            plt.hist(joint_value, color = 'blue', edgecolor = 'black', bins = 100)
            plt.axvline(x=JOINT_LIMITS[joint_name][0], color='r', linewidth=1)
            plt.axvline(x=JOINT_LIMITS[joint_name][1], color='r', linewidth=1)

            # Add labels
            plt.title('Histogram for joint: %s' % joint_name)
            plt.xlabel('Joint revolution')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_dir + "sample_distribution" + '.png')

        """
        Plots for both sampled and original data;
        """
        original_data = pd.read_csv(original_data_file_name, usecols=headers_w_collision)
        print("The keys of the data frame is: %s" % (original_data.keys()))
        original_data_free = original_data[original_data[collision_label] == 1]
        i = 1
        fig = plt.figure(figsize=(10,10))
        for joint_name in JOINT_NAMES:
            plt.subplot(4, 2, i)
            header = 'right_' + joint_name
            original_joint_value = original_data_free[header]
            joint_value = data[header]
            i += 1
            plt.hist([original_joint_value, joint_value], density=True, histtype='bar', color=['lime','blue'],
                     bins = 100, label=['original_values', 'generated_values'])
            plt.axvline(x=JOINT_LIMITS[joint_name][0], color='r', linewidth=1)
            plt.axvline(x=JOINT_LIMITS[joint_name][1], color='r', linewidth=1)
            plt.legend()

            # Add labels
            plt.title('Histogram for joint: %s' % joint_name)
            plt.xlabel('Joint revolution')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_dir + "sample_and_original_distribution" + '.png')

        # TODO: Use visitor pattern to gather statistics on the data;
        # Analysis on the statistics on the data;
        accuracy = data[data[collision_label] == 1].shape[0] / data.shape[0]

        # Compute the KLD between the sample data point and original data points;
        joint_names = ['right_' + joint_name for joint_name in JOINT_NAMES]
        x1 = data[joint_names].values
        x2 = original_data[joint_names].values

        print("The first 10 values of x1 is: %s" % (x1[:10, :]))
        print("The first 10 values of x2 is: %s" % (x2[:10, :]))
        num_bins = np.ones(len(joint_names)).astype(int) * 12 # using 30 degrees in each dimension as resolution;

        limits = np.zeros((len(selected_joints), 2))
        limits[:, 0] = np.minimum(np.min(x1, axis=0), np.min(x2, axis=0))
        limits[:, 1] = np.maximum(np.max(x1, axis=0), np.max(x2, axis=0)) + 1e-9

        print("The limits is: %s" % (limits,))
        print("The number of bins is: %s" % (num_bins,))
        kld = kl_divergence(limits, num_bins, x1, x2)

        spill_over_counts = countSpillover(JOINT_NAMES, data, JOINT_LIMITS)
        total_proportion = countSpilledProportion(JOINT_NAMES, data, JOINT_LIMITS)

        stats = {
            "num_samples": data.shape[0],
            "accuracy": accuracy,
            "kld": kld,
            "spill_over_counts": spill_over_counts,
            "spill_over_proportion": total_proportion
        }

        with open(output_dir + "statistics.json", "w") as fp:
            json.dump(stats, fp)