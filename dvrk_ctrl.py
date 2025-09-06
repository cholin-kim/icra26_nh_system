import crtk
import numpy as np
import sys, os
sys.path.append("/home/surglab/pycharmprojects")
from dvrk_surglab.motion.psm import psm


ral = crtk.ral(node_name='asbasd')
psm1 = psm(ral=ral, arm_name='PSM2')

ral.check_connections()
ral.spin()

# Status 상태 확인후 Enabled로 바꾸기ㅣ.
##https://crtk-robotics.readthedocs.io/en/latest/pages/api.html

print(psm1.measured_js())
print(psm1.is_enabled())
print(psm1.is_busy())

print(psm1.is_disabled())
print(psm1.is_homed())  # homing이 제대로 안됐는데 true 반환함..
# print(psm1.is_fault()) # 이건 없음.

# command를 내리기 전 is_busy()를 확인하고 명령 내리면 될듯. 그런데 움직임이 끝났어도 0 속도로 움직이고 있어  busy 일수도 있음..


# import pdb; pdb.set_trace()
# q_rel = [0.0]*6
# q_rel[3] = 0.05
# psm1.move_jp(np.array(q_rel))
# psm1.disable(10)
psm1.enable(10)
# psm1.home(10)
# psm1.jaw.close()
# psm1.move_jr(np.array([0.0, 0.0, 0.000, 0.0, 0.0, -0.01])).wait(5)
# psm1.move_jp(np.array([ 0.60718972, -0.10149449,  0.15 , -0.6740946 ,  0.34115608,
#         0.39021845])).wait(10)

print(psm1.measured_js())
# print(psm1.measured_cp())

# /home/surglab/anaconda3/envs/minho/bin/python /home/surglab/pycharmprojects/dvrk_surglab/motion/script/PSM_test.py
# [[   -0.123967,    0.961798,    0.244083;
#      -0.89767,   -0.213528,    0.385479;
#      0.422872,   -0.171319,    0.889848]
# [   -0.100835,   0.0253161,    0.156177]]

# psm1.move_cp([])