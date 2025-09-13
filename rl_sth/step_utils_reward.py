import numpy as np
from rl_sth.Kinematics.dvrkKinematics import dvrkKinematics
from rl_sth.Kinematics import dvrkVar

dvrkkin = dvrkKinematics()

def get_obb(joint_positions, Tw_rb):
    '''
    :return: 3 obbb: shaft~wrist, wrist~tool, tool~ee
    '''
    Tbs, _ = dvrkkin.fk(joint_positions)
    Trb_wrist = Tbs[3]
    Trb_pitch = Tbs[4]

    Tw_wrist = Tw_rb @ Trb_wrist
    Tw_pitch = Tw_rb @ Trb_pitch
    Tw_ee = Tw_rb @ Tbs[-1]

    shaft_obb_center = Tw_wrist[:3, 2] * (-0.15) + Tw_wrist[:3, -1]
    # wrist_obb_center = Tw_pitch[:3, 2] * 0.01 + Tw_pitch[:3, -1]
    # gripper_obb_center = Tw_ee[:3, 2] * 0.01 + Tw_ee[:3, -1]
    # wrist_obb_center = Tw_pitch[:3, 2] * 0.005 + Tw_pitch[:3, -1]
    wrist_obb_center = Tw_pitch[:3, 0] * 0.005 + Tw_pitch[:3, -1]
    gripper_obb_center = Tw_ee[:3, 2] * 0.005 + Tw_ee[:3, -1]

    obb_shaft = {}
    obb_shaft['center'] = shaft_obb_center
    obb_shaft['R'] = Tw_wrist[:3, :3]
    obb_shaft['halfWidth'] = np.array([0.004, 0.004, 0.15])

    obb_wrist = {}
    obb_wrist['center'] = wrist_obb_center
    obb_wrist['R'] = Tw_pitch[:3, :3]
    # obb_wrist['halfWidth'] = np.array([0.004, 0.004, 0.01])
    obb_wrist['halfWidth'] = np.array([0.004, 0.004, 0.005])

    obb_gripper = {}
    obb_gripper['center'] = gripper_obb_center
    obb_gripper['R'] = Tw_ee[:3, :3]
    obb_gripper['halfWidth'] = np.array([0.004, 0.004, 0.01])
    obb_gripper['halfWidth'] = np.array([0.004, 0.004, 0.005])

    return obb_shaft, obb_wrist, obb_gripper


def testOBBOBB(center_a, orientation_a, half_lengths_a, center_b, orientation_b, half_lengths_b):
    epsilon = 1e-6
    """
    Return True if two OBBs collide, False otherwise
    """
    Ra_b = np.dot(orientation_a.T, orientation_b)
    ta_b = np.dot(orientation_a.T, center_b - center_a)

    # Absolute of R with epsilon to avoid numerical instability
    abs_R = np.abs(Ra_b) + epsilon

    # Test axes L = A0, L = A1, L = A2
    for i in range(3):
        ra = half_lengths_a[i]
        rb = np.sum(half_lengths_b * abs_R[i, :])

        if np.abs(ta_b[i]) > ra + rb:
            return False

    # Test axes L = B0, L = B1, L = B2
    for i in range(3):
        ra = np.sum(half_lengths_a * abs_R[:, i])
        rb = half_lengths_b[i]
        if np.abs(np.dot(ta_b, Ra_b[:, i])) > ra + rb:
            return False

    # Test axis L = A0 x B0
    ra = half_lengths_a[1] * abs_R[2, 0] + half_lengths_a[2] * abs_R[1, 0]
    rb = half_lengths_b[1] * abs_R[0, 2] + half_lengths_b[2] * abs_R[0, 1]
    if np.abs(ta_b[2] * Ra_b[1, 0] - ta_b[1] * Ra_b[2, 0]) > ra + rb:
        return False

    # Test axis L = A0 x B1
    ra = half_lengths_a[1] * abs_R[2, 1] + half_lengths_a[2] * abs_R[1, 1]
    rb = half_lengths_b[0] * abs_R[0, 2] + half_lengths_b[2] * abs_R[0, 0]
    if np.abs(ta_b[2] * Ra_b[1, 1] - ta_b[1] * Ra_b[2, 1]) > ra + rb:
        return False

    # Test axis L = A0 x B2
    ra = half_lengths_a[1] * abs_R[2, 2] + half_lengths_a[2] * abs_R[1, 2]
    rb = half_lengths_b[0] * abs_R[0, 1] + half_lengths_b[1] * abs_R[0, 0]
    if np.abs(ta_b[2] * Ra_b[1, 2] - ta_b[1] * Ra_b[2, 2]) > ra + rb:
        return False

    # Test axis L = A1 x B0
    ra = half_lengths_a[0] * abs_R[2, 0] + half_lengths_a[2] * abs_R[0, 0]
    rb = half_lengths_b[1] * abs_R[1, 2] + half_lengths_b[2] * abs_R[1, 1]
    if np.abs(ta_b[0] * Ra_b[2, 0] - ta_b[2] * Ra_b[0, 0]) > ra + rb:
        return False

    # Test axis L = A1 x B1
    ra = half_lengths_a[0] * abs_R[2, 1] + half_lengths_a[2] * abs_R[0, 1]
    rb = half_lengths_b[0] * abs_R[1, 2] + half_lengths_b[2] * abs_R[1, 0]
    if np.abs(ta_b[0] * Ra_b[2, 1] - ta_b[2] * Ra_b[0, 1]) > ra + rb:
        return False

    # Test axis L = A1 x B2
    ra = half_lengths_a[0] * abs_R[2, 2] + half_lengths_a[2] * abs_R[0, 2]
    rb = half_lengths_b[0] * abs_R[1, 1] + half_lengths_b[1] * abs_R[1, 0]
    if np.abs(ta_b[0] * Ra_b[2, 2] - ta_b[2] * Ra_b[0, 2]) > ra + rb:
        return False

    # Test axis L = A2 x B0
    ra = half_lengths_a[0] * abs_R[1, 0] + half_lengths_a[1] * abs_R[0, 0]
    rb = half_lengths_b[1] * abs_R[2, 2] + half_lengths_b[2] * abs_R[2, 1]
    if np.abs(ta_b[1] * Ra_b[0, 0] - ta_b[0] * Ra_b[1, 0]) > ra + rb:
        return False

    # Test axis L = A2 x B1
    ra = half_lengths_a[0] * abs_R[1, 1] + half_lengths_a[1] * abs_R[0, 1]
    rb = half_lengths_b[0] * abs_R[2, 2] + half_lengths_b[2] * abs_R[2, 0]
    if np.abs(ta_b[1] * Ra_b[0, 1] - ta_b[0] * Ra_b[1, 1]) > ra + rb:
        return False

    # Test axis L = A2 x B2
    ra = half_lengths_a[0] * abs_R[1, 2] + half_lengths_a[1] * abs_R[0, 2]
    rb = half_lengths_b[0] * abs_R[2, 1] + half_lengths_b[1] * abs_R[2, 0]
    if np.abs(ta_b[1] * Ra_b[0, 2] - ta_b[0] * Ra_b[1, 2]) > ra + rb:
        return False

    # Since no separating axis is found, the OBBs must be intersecting
    return True

class RewardUtils:

    @staticmethod
    def check_collision(Tw_ntarg1, Tw_ntarg2):
        '''
        :return: True asap if any collision occurs, else False
        '''
        Trb1_ed1 = np.linalg.inv(env_config['Tw_rb1']) @ Tw_ntarg1
        Trb2_ed2 = np.linalg.inv(env_config['Tw_rb2']) @ Tw_ntarg2

        targ_q1 = dvrkkin.ik(Trb1_ed1)
        targ_q2 = dvrkkin.ik(Trb2_ed2)

        obb_shaft1, obb_wrist1, obb_gripper1 = get_obb(targ_q1, env_config['Tw_rb1'])
        obb_shaft2, obb_wrist2, obb_gripper2 = get_obb(targ_q2, env_config['Tw_rb2'])

        group_1 = [obb_shaft1, obb_wrist1, obb_gripper1]
        group_2 = [obb_shaft2, obb_wrist2, obb_gripper2]

        for a in group_1:
            for b in group_2:
                collision_test = testOBBOBB(center_a=a['center'], orientation_a=a['R'],
                                            half_lengths_a=a['halfWidth'], center_b=b['center'],
                                            orientation_b=b['R'], half_lengths_b=b['halfWidth'])
                if collision_test:
                    return True
        return False

    ### Sihyeoung edited
    @staticmethod
    def check_collision_ground(T_link1, T_link2, ground_threshold, rb_z):  # [4x4 * 8], [4x4 *8]
        '''
        Returns: True asap if any collision occurs, else False
        '''
        T_link1_z = [T[2, 3] for T in T_link1]
        T_link2_z = [T[2, 3] for T in T_link2]
        # 예: ground collision 체크
        for i, z in enumerate(T_link1_z):
            # if z+rb_z < -ground_threshold:
            if z + 0.145 < -ground_threshold:
                return True
            else:
                pass

        for i, z in enumerate(T_link2_z):
            # if z+rb_z < -ground_threshold:
            if z + 0.145 < -ground_threshold:
                return True
            else:
                pass

        return False
    ###

    @staticmethod
    def joint_isin_limit(joint_pos):
        '''
        :return: True if joint position at limit
        '''
        return np.any(joint_pos < dvrkVar.joint_range_lower_limit) or \
            np.any(joint_pos > dvrkVar.joint_range_upper_limit)

if __name__ == "__main__":
    utils_rewaard = RewardUtils
    Tw_ntarg1 = np.array(
    [[8.66025404e-01 ,- 5.00000000e-01 ,- 1.22460635e-16 , 4.15692194e-02],
     [-5.00000000e-01 ,- 8.66025404e-01 , 1.49966072e-32 , 7.60000000e-02],
    [-1.06054021e-16,
    6.12303177e-17, - 1.00000000e+00,
    5.00000000e-02],
    [0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])
    Tw_ntarg2=np.array(
    [[-9.95839695e-17, - 1.00000000e+00 , 1.22460635e-16, - 2.93895818e-18],
     [1.00000000e+00, - 9.95839695e-17, - 1.22460635e-16 , 5.20000000e-02],
    [1.22460635e-16,
    1.22460635e-16,
    1.00000000e+00,
    5.00000000e-02],
    [0.00000000e+00,  0.00000000e+00  ,0.00000000e+00  ,1.00000000e+00]])

    from pybullet_collision.obb_vis import collision_pybullet_dual
    pyb = collision_pybullet_dual(headless=False)
    import pybullet as p
    from _config import env_config
    Trb1_ree1 = np.linalg.inv(env_config.Tw_rb1) @ Tw_ntarg1
    Trb2_ree2 = np.linalg.inv(env_config.Tw_rb2) @ Tw_ntarg2
    t=0
    while True:
        joint_config1 = dvrkkin.ik(Trb1_ree1)
        joint_config2 = dvrkkin.ik(Trb2_ree2)

        joint_config = np.concatenate([joint_config1, joint_config2])
        pyb.set_joint_positions_dual(joint_config)

        Tw_ree1 = env_config['Tw_rb1'] @ Trb1_ree1
        Tw_ree2 = env_config['Tw_rb2'] @ Trb2_ree2

        pyb.set_joint_positions_dual(joint_config)

        for _ in range(100):
            p.stepSimulation()

        cur_joint1 = pyb.get_joint_positions(robot_id=pyb.robot_id1)
        cur_joint2 = pyb.get_joint_positions(robot_id=pyb.robot_id2)

        flag1 = np.linalg.norm(cur_joint1 - joint_config1) < 1e-2
        flag2 = np.linalg.norm(cur_joint2 - joint_config2) < 1e-2
        try:
            flag1 and flag2
        except:
            print("failed to reach joint")

        obb_shaft1, obb_wrist1, obb_gripper1 = get_obb(joint_config1, env_config['Tw_rb1'])
        obb_shaft2, obb_wrist2, obb_gripper2 = get_obb(joint_config2, env_config['Tw_rb2'])

        obb_shaft1_id = pyb.draw_obb(obb_shaft1, rgba_color=[0.5, 0, 1, 0.5])
        obb_shaft2_id = pyb.draw_obb(obb_shaft2, rgba_color=[0.5, 0, 1, 0.5])
        obb_wrist1_id = pyb.draw_obb(obb_wrist1, rgba_color=[0, 1, 0, 0.5])
        obb_wrist2_id = pyb.draw_obb(obb_wrist2, rgba_color=[0, 1, 0, 0.5])
        obb_gripper1_id = pyb.draw_obb(obb_gripper1, rgba_color=[0, 1, 0.5, 0.5])
        obb_gripper2_id = pyb.draw_obb(obb_gripper2, rgba_color=[0, 1, 0.5, 0.5])

        # 3초간 시뮬레이션
        # for _ in range(1 * 240):  # 240Hz 기준
        #     p.stepSimulation()
        #     time.sleep(1. / 240)
        p.removeBody(obb_shaft1_id)
        p.removeBody(obb_shaft2_id)
        p.removeBody(obb_wrist1_id)
        p.removeBody(obb_wrist2_id)
        p.removeBody(obb_gripper1_id)
        p.removeBody(obb_gripper2_id)

        collision_obb = utils_rewaard.check_collision(Tw_ree1, Tw_ree2)
        if collision_obb:
            print("OBB collision")
        if pyb.check_collision():
            print("Collision detected.")
        else:
            # print("No Collision.")
            pass

        # update parameter
        t += 1

    p.disconnect()
    print(utils_rewaard.check_collision(Tw_ntarg1, Tw_ntarg2))