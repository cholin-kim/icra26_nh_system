import crtk
import numpy as np
import sys, os
sys.path.append("/home/surglab/pycharmprojects")
from dvrk_surglab.motion.psm import psm
import time
import math
from Kinematics.dvrkKinematics import dvrkKinematics
from scipy.spatial.transform import Rotation as R, Slerp


class dvrkCmd():
    def __init__(self):
        self.ral = crtk.ral(node_name='dvrkCtrl')
        self.ral.check_connections()
        self.ral.spin()

        self.psm1 = psm(ral=self.ral, arm_name='PSM1')
        self.psm2 = psm(ral=self.ral, arm_name='PSM2')

        if self.psm1.is_disabled():
            self.psm1.enable()
        if self.psm2.is_disabled():
            self.psm2.enable()

        self.home()



    def home(self):
        self.ral.check_connections()

        print('starting enable')
        if not self.psm1.enable(5):
            sys.exit('PSM1|failed to enable within 10 seconds')
        if not self.psm2.enable(5):
            sys.exit('PSM2|failed to enable within 10 seconds')
        print('starting home')
        if not self.psm1.home(5):
            sys.exit('PSM1|failed to home within 10 seconds')
        if not self.psm2.home(5):
            sys.exit('PSM2|failed to home within 10 seconds')

        # # get current joints just to set size
        # print('move to starting position')
        # goal = numpy.copy(self.arm.setpoint_jp())
        # # go to zero position, make sure 3rd joint is past cannula
        # goal.fill(0)
        # goal[2] = 0.12
        # self.arm.move_jp(goal).wait()

    def get_joint_positions(self, which='PSM1'):
        if which == 'PSM1':
            joint_state = self.psm1.measured_js()
        elif which == 'PSM2':
            joint_state = self.psm2.measured_js()
        return joint_state[0]

    def get_ee_pose(self, which='PSM1'):
        joint_state = self.get_joint_positions(which)
        Tb2ee = dvrkKinematics.fk(joint_state)[0][-1]
        return Tb2ee

    def interpolate_cartesian_pose(self, T_start, T_end, steps=50):
        # Extract translations
        p_start = T_start[:3, 3]
        p_end = T_end[:3, 3]

        # Extract rotations
        R_start = R.from_matrix(T_start[:3, :3])
        R_end = R.from_matrix(T_end[:3, :3])

        # Prepare slerp
        key_times = [0, 1]
        key_rots = R.from_matrix([T_start[:3, :3], T_end[:3, :3]])
        slerp = Slerp(key_times, key_rots)

        # Interpolate
        poses = []
        for t in np.linspace(0, 1, steps):
            p_interp = (1 - t) * p_start + t * p_end
            R_interp = slerp([t])[0]

            # Compose homogeneous transform
            T_interp = np.eye(4)
            T_interp[:3, :3] = R_interp.as_matrix()
            T_interp[:3, 3] = p_interp
            poses.append(T_interp)

        return poses

    def set_pose(self, Tb2ee_targ=np.identity(4), which='PSM1'):
        Tb2ee_cur = self.get_ee_pose(which)
        Ts = self.interpolate_cartesian_pose(T_start=Tb2ee_cur, T_end=Tb2ee_targ, steps=20)
        for T in Ts:
            q = dvrkKinematics.ik(T)
            if which=='PSM1':
                self.psm1.move_jp(q).wait()
            elif which=='PSM2':
                self.psm2.move_jp(q).wait()
            time.sleep(0.04)




    def set_joint_positions_rel(self, q_targ, which='PSM1'):
        rad_step = 0.03
        m_step = 0.003
        rad_threshold = rad_step / 4
        m_threshold = m_step / 4
        step = np.array([rad_step, rad_step, m_step, rad_step, rad_step, rad_step])
        threshold = np.array([rad_threshold, rad_threshold, m_threshold, rad_threshold, rad_threshold, rad_threshold])

        # if which == 'PSM1':
        #     jaw = self.psm1.jaw.measured_js()[0]
        # elif which == 'PSM2':
        #     jaw = self.psm1.jaw.measured_js()[0]

        # q_targ[-1] += jaw


        while True:
            q_cur = self.get_joint_positions(which=which)
            # print("q_cur:", q_cur)
            if (np.abs(q_targ - q_cur) < threshold).all():
                print("Reached target")
                break

            diff = q_targ - q_cur
            q_rel = np.sign(diff) * np.minimum(np.abs(diff), step)
            # print("q_rel:", q_rel)
            if which == 'PSM1':
                self.psm1.move_jr(q_rel).wait()
            elif which == 'PSM2':
                self.psm2.move_jr(q_rel).wait()
            time.sleep(0.01)

    def open_jaw(self, which='PSM1', angle=60.0):
        if which == 'PSM1':
            if self.psm1.jaw.measured_js()[0] > 1.0:
                pass
            else:
                # self.psm1.jaw.move_jp(np.array([math.radians(40)]))
                self.psm1.jaw.open(angle=math.radians(angle)).wait()
        elif which == 'PSM2':
            if self.psm2.jaw.measured_js()[0] > 1.0:
                pass
            else:
                # self.psm2.jaw.open()
                self.psm2.jaw.open(angle=math.radians(angle)).wait()

    def close_jaw(self, which='PSM1'):
        if which == 'PSM1':
            self.psm1.jaw.close().wait()
        elif which == 'PSM2':
            self.psm2.jaw.close().wait()

    def keyboard_servo(self, key, which='PSM1'):
        cur_q = self.get_joint_positions(which=which)
        Trb_ee = dvrkKinematics.fk(cur_q)[0][-1]
        if key == 'UP':
            self.up(Trb_ee, which=which)
        elif key == 'DOWN':
            self.down(Trb_ee, which=which)
        elif key == 'LEFT':
            self.left(Trb_ee, which=which)
        elif key == 'RIGHT':
            self.right(Trb_ee, which=which)
        elif key == 'FORWARD':
            self.forward(Trb_ee, which=which)
        elif key == 'BACKWARD':
            self.backward(Trb_ee, which=which)

    def up(self, Trb_ee, which='PSM1'):
        Tee_ee = np.identity(4)
        Tee_ee[2, -1] = 0.002
        Trb_ee = Trb_ee @ Tee_ee
        self.set_pose(Trb_ee, which=which)

    def down(self, Trb_ee, which='PSM1'):
        Tee_ee = np.identity(4)
        Tee_ee[2, -1] = -0.002
        Trb_ee = Trb_ee @ Tee_ee
        self.set_pose(Trb_ee, which=which)

    def right(self, Trb_ee, which='PSM1'):
        Tee_ee = np.identity(4)
        Tee_ee[0, -1] = 0.004
        Trb_ee = Trb_ee @ Tee_ee
        self.set_pose(Trb_ee, which=which)

    def left(self, Trb_ee, which='PSM1'):
        Tee_ee = np.identity(4)
        Tee_ee[0, -1] = -0.004
        Trb_ee = Trb_ee @ Tee_ee
        self.set_pose(Trb_ee, which=which)

    def forward(self, Trb_ee, which='PSM1'):
        Tee_ee = np.identity(4)
        Tee_ee[1, -1] = 0.004
        Trb_ee = Trb_ee @ Tee_ee
        self.set_pose(Trb_ee, which=which)

    def backward(self, Trb_ee, which='PSM1'):
        Tee_ee = np.identity(4)
        Tee_ee[1, -1] = -0.004
        Trb_ee = Trb_ee @ Tee_ee
        self.set_pose(Trb_ee, which=which)

    def set_joint_rel(self, q_targ, which='PSM1'):
        rad_step = 0.02
        m_step = 0.002
        rad_threshold = rad_step / 4
        m_threshold = m_step / 4
        step = np.array([rad_step, rad_step, m_step, rad_step, rad_step, rad_step])
        threshold = np.array([rad_threshold, rad_threshold, m_threshold, rad_threshold, rad_threshold, rad_threshold])
        q_cur = self.get_joint_positions(which=which)

        if (np.abs(q_targ - q_cur) < threshold).all():
            print("Reached target")
            return

        diff = q_targ - q_cur
        # q_rel = np.sign(diff) * np.minimum(np.abs(diff), step)
        print("diff:", diff)

        if which == 'PSM1':
            self.psm1.move_jr(diff).wait()
        elif which == 'PSM2':
            self.psm2.move_jr(diff).wait()




if __name__ == "__main__":
    cmd = dvrkCmd()
    psm_blue = 'PSM2'
    psm_yellow = 'PSM1'
    cmd.open_jaw(which=psm_yellow)
    cmd.open_jaw(which=psm_blue)
    quit()
    # cmd.close_jaw(which=psm_blue)
    # cmd.close_jaw(which=psm_yellow)
    Tcam_rbBlue = np.array(
        [[0.958227689670898, -0.21768249007965917, -0.18551018371154965, 0.17551387734934729],
         [-0.09430895426216324, 0.37185710016414775, -0.9234869345061079, -0.17092286002679188],
         [0.27001021442521606, 0.9024060231238654, 0.33579436197741447, 0.19878618409315504], [0.0, 0.0, 0.0, 1.0]]

    )
    Tcam_rbYellow = np.array(
        [[0.9977628695816074, -0.04424886682120791, -0.05011281142896951, -0.1929004230309927],
         [-0.03362939432434292, 0.3156440822984118, -0.9482815389679223, -0.16374099581128818],
         [0.057778195901692066, 0.9478453729881434, 0.31344988272978047, 0.14549715406600355], [0.0, 0.0, 0.0, 1.0]]
    )
    q = cmd.get_joint_positions(which=psm_blue)
    # print(dvrkKinematics.fk(q)[0][-1].tolist())
    print(Tcam_rbBlue @ dvrkKinematics.fk(q)[0][-1])
    Tcam_w = np.identity(4)
    rvec = [2.000704471390556, -0.004855342843570476, -2.9530378153928445]
    Tcam_w[:3, -1] = [0.005541766210923305, -0.0103154460209131, 0.14533409657221247]
    Tcam_w[:3, :3] = R.from_euler('XYZ', rvec).as_matrix()
    Tw_cam = np.linalg.inv(Tcam_w)
    print(Tw_cam @ Tcam_rbBlue @ dvrkKinematics.fk(q)[0][-1])
    q = cmd.get_joint_positions(which=psm_yellow)
    # print(dvrkKinematics.fk(q)[0][-1].tolist())
    print(Tcam_rbYellow @ dvrkKinematics.fk(q)[0][-1])
    quit()
    # cmd.open_jaw(which=psm_blue)

    # T = cmd.get_ee_pose(psm_blue)
    # T_targ = T.copy()
    # T_targ[:2, -1] += 0.01
    # cmd.set_pose(T_targ, which=psm_blue)

    quit()



    targ = cmd.get_joint_positions(which=psm_yellow)
    targ[-2] -= 0.1
    cmd.set_joint_positions_rel(targ, which=psm_yellow)

    import cv2
    def get_key_input():
        """
        이미지(img)를 띄우면서 화살표키 / WASD / q 입력 대기
        q: 종료
        반환: 'UP', 'DOWN', 'LEFT', 'RIGHT', 'QUIT' 또는 None
        """
        img = np.zeros((1, 1))
        cv2.imshow("get_key_input", img)
        print("키 입력 대기 중... (↑/↓/←/→ 또는 WASD, q 종료)")

        while True:
            key = cv2.waitKey(0) & 0xFF  # 0 → 입력까지 대기
            if key in (82, ord('w'), ord('W')):
                return 'UP'
            elif key in (84, ord('s'), ord('S')):
                return 'DOWN'
            elif key in (81, ord('a'), ord('A')):
                return 'LEFT'
            elif key in (83, ord('d'), ord('D')):
                return 'RIGHT'
            elif key in (ord('q'), ord('Q')):
                return 'QUIT'
            elif key in (ord('f'), ord('F')):
                return 'FORWARD'
            elif key in (ord('b'), ord('B')):
                return 'BACKWARD'
            else:
                print(f"Unknown key: {key}")


    key = get_key_input()
    while not key == 'QUIT':
        cmd.keyboard_servo(key, which='PSM1')
        key = get_key_input()

