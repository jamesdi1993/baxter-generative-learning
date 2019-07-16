from src.utils import get_path, parse_args, find_data_file
from src.path_config import ROS_BASE_PATH, VALIDATED_OUTPUT_TEMPLATE, OUTPUT_BASE_PATH

import argparse
import os
import shutil

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--env')
    parser.add_argument('--run-id')
    parser.add_argument('--dry-run', action='store_true')

    args, _ = parser.parse_known_args()
    env = args.env
    run_id = args.run_id
    dry_run = args.dry_run

    src_file = os.path.join(ROS_BASE_PATH, VALIDATED_OUTPUT_TEMPLATE % (env, run_id))
    dst_file = VALIDATED_OUTPUT_TEMPLATE % (env, run_id)
    dst_dir = OUTPUT_BASE_PATH % (env, run_id)

    print("Copying file from %s to %s" % (src_file, dst_file))
    if not dry_run:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        dst_file = shutil.copy(src_file, dst_dir)
        print("Copied file from src to destination.")
