#!/bin/bash
beta_array=(0.05)
num_joints=7

# Activate Python3 env;
. activate baxter_vae_collision
# Train and make inference;
for beta in "${beta_array[@]}"
do
    # python vae_baxter_self_collision.py --beta $beta --d-input $num_joints --epochs 10 --generated-sample-size 10000000
    # printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

    # copy from workspace to ros to run collision check
    python copy_output_to_ros.py --beta $beta --num-joints $num_joints
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

    conda deactivate
    python /home/nikhildas/ros_ws/src/baxter_moveit_config/src/joint_state_filler.py --num-joints $num_joints --beta $beta
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -

    # Run analysis on the new data;
    . activate baxter_vae_self_collision
    python copy_validated_output_from_ros.py --beta $beta --num-joints $num_joints
    python self_collision_data_analysis.py --beta $beta --num-joints $num_joints
    # cut -d, -f8 ../data/validated_output/7_num_joints/"$beta"_beta/right_7_1000000_filled.csv | tail -n+2 | sort | uniq -c \
    # > ../data/results/7_num_joints/"$beta"_beta/statistics.txt
done