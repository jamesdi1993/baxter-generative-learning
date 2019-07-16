from src.utils import get_path
from src.path_config import SAMPLED_OUTPUT_TEMPLATE, ROS_BASE_PATH, OUTPUT_BASE_PATH

import argparse
import os
import shutil

# TODO: Merge this with copy_validated_out_from_ros script, since they are really doing the same thing.
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env')
    parser.add_argument('--run-id')
    parser.add_argument('--dry-run', action='store_true')

    args, _ = parser.parse_known_args()
    env = args.env
    run_id = args.run_id
    dry_run = args.dry_run

    src_file = SAMPLED_OUTPUT_TEMPLATE % (env, run_id)
    dst_dir = os.path.join(ROS_BASE_PATH, OUTPUT_BASE_PATH % (env, run_id) )
    dst_file = os.path.join(ROS_BASE_PATH, src_file)

    print("Copying file: %s to : %s" % (src_file, dst_file))
    if not dry_run:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        shutil.copy(src_file, dst_file)
        print("Copied file from src to destination.")
