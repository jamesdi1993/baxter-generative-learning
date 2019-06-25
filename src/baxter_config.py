import numpy as np

JOINT_NAMES = ['s0','s1','e0','e1','w0','w1','w2']
JOINT_LIMITS = {
    's0': [-1.7016, 1.7016],
    's1': [-2.147, 1.047],
    'e0': [-3.0541, 3.0541],
    'e1': [-0.05, 2.618],
    'w0': [-3.059, 3.059],
    'w1': [-1.5707, 2.094],
    'w2': [-3.059, 3.059]
}

COLLISION_KEY = 'collisionFree'

def get_limb_headers(limb_name):
    return [limb_name + "_" + joint_name for joint_name in JOINT_NAMES]

def get_joint_names(joints):
    return ['right_' + joint_name for joint_name in joints]

def get_headers_with_collision(joints):
    headers = get_joint_names(joints)
    headers.append(COLLISION_KEY)
    return headers

def get_joint_limits(joints):
    joint_limits = [JOINT_LIMITS[joint] for joint in joints]
    joint_limits = np.array(joint_limits).T
    return joint_limits