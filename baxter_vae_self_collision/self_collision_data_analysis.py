#!/usr/bin/env python
# coding: utf-8
from baxter_config import JOINT_LIMITS, JOINT_NAMES, get_headers_with_collision, COLLISION_KEY
from utils import find_data_file, get_path, parse_args
from path_config import result_base_path, input_base_path, validated_output_base_path

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

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
    args, _ = parser.parse_known_args()

    num_joints = args.num_joints

    path_args = parse_args(args)
    validated_output_path = get_path(validated_output_base_path, path_args)
    file_name = find_data_file(validated_output_path, num_joints)
    original_data_file_name = find_data_file(input_base_path, num_joints)

    output_dir = get_path(result_base_path, path_args)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    selected_joints = JOINT_NAMES[0:num_joints]
    headers_w_collision = get_headers_with_collision(selected_joints)
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
    original_data_free = original_data[original_data['collisionFree'] == 1]
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
    accuracy = data[data[COLLISION_KEY] == 1].shape[0] / data.shape[0]
    spill_over_counts = countSpillover(JOINT_NAMES, data, JOINT_LIMITS)
    total_proportion = countSpilledProportion(JOINT_NAMES, data, JOINT_LIMITS)

    stats = {
        "num_samples": data.shape[0],
        "accuracy": accuracy,
        "spill_over_counts": spill_over_counts,
        "spill_over_proportion": total_proportion
    }

    with open(output_dir + "statistics.json", "w") as fp:
        json.dump(stats, fp)