import time

INPUT_BASE_PATH = "./data/%s/input/"
TEST_OUTPUT_BASE_PATH = "./data/test_output/"
ROS_BASE_PATH = "/home/nikhildas/ros_ws/src/baxter_moveit_config/"

OUTPUT_BASE_PATH = "./data/%s/output/%s/"
SAMPLED_OUTPUT_TEMPLATE = "./data/%s/output/%s/generated_output.csv"
VALIDATED_OUTPUT_TEMPLATE = "./data/%s/output/%s/validated.csv"

PATH_ARGS = ["num_joints", "beta"]

def get_run_id(args):
    return get_run_prefix(args) + '/' + str(int(time.time())) # Each run should be assigned an unique id;

def get_run_prefix(args):
    """
    Get the run id for a given config;
    :param args:
    :return:
    """
    return "joints%s_dinput%s_dlatent%s_epochs%s_beta%s" % (
        args.num_joints,
        args.d_input,
        args.d_output,
        args.epochs,
        args.beta
    )