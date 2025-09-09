from shutil import which

import crtk
import numpy as np
import sys, os
sys.path.append("/home/surglab/pycharmprojects")
from dvrk_surglab.motion.psm import psm
import time

ral = crtk.ral(node_name='dvrkCtrl')

class dvrkCmd():
    def __init__(self):
        ral.check_connections()
        ral.spin()

        self.psm1 = psm(ral=ral, arm_name='PSM1')
        self.psm2 = psm(ral=ral, arm_name='PSM2')

        if self.psm1.is_disabled():
            self.psm1.enable()
        if self.psm2.is_disabled():
            self.psm2.enable()
        # is_homed()
        # is_busy()
        self.psm1.jaw.open()
        self.psm2.jaw.open()

    def get_joint_positions(self, which='PSM1'):
        if which == 'PSM1':
            joint_state = self.psm1.measured_js()
        elif which == 'PSM2':
            joint_state = self.psm2.measured_js()
        return joint_state[0]

    def set_joint_positions_rel(self, q_targ, which='PSM1'):
        rad_step = 0.02
        m_step = 0.002
        rad_threshold = rad_step / 2
        m_threshold = m_step / 2
        step = np.array([rad_step, rad_step, m_step, rad_step, rad_step, rad_step])
        threshold = np.array([rad_threshold, rad_threshold, m_threshold, rad_threshold, rad_threshold, rad_threshold])

        while True:
            q_cur = self.get_joint_positions(which=which)
            if (np.abs(q_targ - q_cur) < threshold).all():
                print("Reached target")
                break

            diff = q_targ - q_cur
            q_rel = np.sign(diff) * np.minimum(np.abs(diff), step)
            if which == 'PSM1':
                self.psm1.move_jr(q_rel)
            elif which == 'PSM2':
                self.psm2.move_jr(q_rel)
            time.sleep(3)

if __name__ == "__main__":
    cmd = dvrkCmd()
    robot1 = 'PSM1'
    robot2 = 'PSM2'
    joint_pos1 = cmd.get_joint_positions(which=robot1)
    print(joint_pos1)
    targ_joint_pos = joint_pos1 + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.05])
    cmd.set_joint_positions_rel(targ_joint_pos, which=robot1)
    print(cmd.get_joint_positions(which=robot1))