import time

input_base_path = "./data/%s/input/"
output_base_path = "./data/%s/output/"
validated_output_base_path = "./data/%s/validated_output/"
result_base_path = "./data/%s/results/"
model_base_path = "./data/%s/model/"
test_output_base_path = "./data/test_output/"
ros_base_path = "/home/nikhildas/ros_ws/src/baxter_moveit_config/data/generated_data/"

SAMPLED_OUTPUT_TEMPLATE = "./data/%s/output/%s/generated_output.csv"
VALIDATED_OUTPUT_TEMPLATE = "./data/%s/output/%s/validated.csv"

PATH_ARGS = ["num_joints", "beta"]

def get_run_id(args):
    return get_run_prefix(args) + '/' + time.time() # Each run should be assigned an unique id;

def get_run_prefix(args):
    """
    Get the run id for a given config;
    :param args:
    :return:
    """
    return "joints%s_dinput%s_dlatent%s_doutput%s_epochs%s_beta%s" % (
        args.num_joints,
        args.d_input,
        args.d_latent,
        args.d_output,
        args.epochs,
        args.beta
    )