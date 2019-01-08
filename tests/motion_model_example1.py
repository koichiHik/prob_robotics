
import sys
import os
sys.path.append(os.pardir)

import math

# Common Module
from common.container import Pose2D

# Motion Model
from motion_model.motion_model import MotionErrorModel2dConfigParams
from motion_model.motion_model import MotionErrorModel2D

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    print("Sample Motion")

    path = [Pose2D(0.0, 0.0, 0.0), Pose2D(0.0, 0.0, 0.0), Pose2D(1.0, 0.0, 0.0), \
            Pose2D(2.0, 0.0, 0.0), Pose2D(3.0, 0.0, 0.0), Pose2D(4.0, 0.0, 0.0), \
            Pose2D(5.0, 0.0, 0.0), Pose2D(5.0, 0.0, math.pi/2.0), \
            Pose2D(5.0, 1.0, math.pi/2.0), Pose2D(5.0, 2.0, math.pi/2.0), \
            Pose2D(5.0, 2.0, math.pi/1.0), Pose2D(4.0, 2.0, math.pi/1.0), \
            Pose2D(3.0, 2.0, math.pi/1.0), Pose2D(2.0, 2.0, math.pi/1.0), \
            Pose2D(1.0, 2.0, math.pi/1.0), Pose2D(0.0, 2.0, math.pi/1.0)]

    # Error Model
    err_conf = MotionErrorModel2dConfigParams(std_rot_per_rot=0.01,
                                              std_rot_per_trans=0.02,
                                              std_trans_per_trans=0.05,
                                              std_trans_per_rot=0.01)
    err_model = MotionErrorModel2D(conf=err_conf)

    particle_num = 5  
    plt.scatter(x=path[0].x,y=path[0].y,s=2)

    cur_particle_poses = \
        [Pose2D(path[0].x, path[0].y, path[0].theta) for i in range(particle_num)]

    for i in range(1, 16):
      cur_particle_poses = \
            err_model.sample_motion(cur_odom=path[i], \
                                    last_odom=path[i-1], \
                                    last_particle_poses=cur_particle_poses)

      plt.scatter(x=[pose.x for pose in cur_particle_poses], \
                  y=[pose.y for pose in cur_particle_poses], s=2)
    
    plt.show()
