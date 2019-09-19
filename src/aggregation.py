import argparse
import numpy as np
import pandas as pd

positions = ['x', 'y', 'z']
collsion_keys = ['self_collision_free', 'one_box_environment_collision_free']

def main(args):
    file_name = args.file_path
    res = args.resolution
    limits = 1.5 * np.array([-1, 1, -1, 1, -1, 1]) + np.array([0.5, 0.5, 0, 0, 0, 0])
    print('limits.shape: %s' % (limits.shape,))
    xs = np.ceil((limits[1] - limits[0]) / res).astype(int)
    ys = np.ceil((limits[3] - limits[2]) / res).astype(int)
    zs = np.ceil((limits[5] - limits[4]) / res).astype(int)
    total = np.zeros((xs, ys, zs))
    self_collsion = np.zeros(total.shape)
    env_collision = np.zeros(total.shape)


    # read from csv file;
    data = pd.read_csv(file_name, usecols= positions + collsion_keys)
    pos = data.loc[:, positions].values
    indices = np.floor((pos - limits[[0, 2, 4]]) / res).astype(int)
    for i in range(indices.shape[0]):
        total[indices[i, 0], indices[i, 1], indices[i, 2]] += 1
        if data.loc[i, 'self_collision_free'] == 0:
            self_collsion[indices[i, 0], indices[i, 1], indices[i, 2]] += 1
        if data.loc[i, 'one_box_environment_collision_free'] == 1:
            env_collision[indices[i, 0], indices[i, 1], indices[i, 2]] += 1

    print("Total self collision free percentage: %s" % (np.sum(self_collsion)/np.sum(total)))
    print("Total env collision free percentage: %s" % (np.sum(env_collision)/np.sum(total)))
    collision_p = np.nan_to_num(env_collision / total)
    output = args.output_path
    np.save(output, collision_p)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path')
    parser.add_argument('--output-path')
    parser.add_argument('--resolution', type=float)

    args, _ = parser.parse_known_args()
    main(args)
