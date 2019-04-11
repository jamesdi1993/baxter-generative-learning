#!/usr/bin/env python
# coding: utf-8
from baxter_config import JOINT_LIMITS, JOINT_NAMES, get_headers_with_collision
from utils import find_data_file, get_path, parse_args
from path_config import result_base_path, input_base_path, validated_output_base_path

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

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
    original_data_free = original_data[original_data['inCollision'] == 1]
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



