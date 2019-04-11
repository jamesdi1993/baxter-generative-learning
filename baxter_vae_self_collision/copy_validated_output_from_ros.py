from utils import get_path, parse_args, find_data_file
from path_config import ros_base_path, validated_output_base_path

import argparse
import os
import shutil

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--num-joints', type = int, default= 7)

    # Fixed static parameters;
    parser.add_argument('--h-dim1', type=int, default=256)
    parser.add_argument('--h-dim2', type=int, default=100)
    parser.add_argument('--d-output', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--use-cuda', type=bool, default=False)
    parser.add_argument('--generated-sample-size', type=bool, default=1000000)

    args, _ = parser.parse_known_args()

    # Find the path to write to;
    path_args = parse_args(args)

    input_dir = get_path(ros_base_path + "validated/", path_args)
    src_file = find_data_file(input_dir, args.num_joints)
    dst_dir = get_path(validated_output_base_path, path_args)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    dst_file = shutil.copy(src_file, dst_dir)

    print("Copied file from src to destination.")
    print("Src: " + src_file)
    print("Dst: " + dst_file)
