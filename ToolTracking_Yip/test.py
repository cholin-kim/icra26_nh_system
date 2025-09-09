import numpy as np
import cv2


# a = np.array([[ 0.02365263, -0.99941224 , 0.02481347, -0.26439474],
#  [-0.38888067 ,-0.0320634,  -0.92072999 , 0.17925684],
#  [ 0.92098445,  0.01212821, -0.38941047 ,-0.141734  ],
#  [ 0.      ,    0.     ,     0.       ,   1.        ]])


# a = np.linalg.inv(a)

# np.save('T_base2cam4.npy', a)   



# img = np.array([
#     [ [255, 0, 255] ]
# ], dtype=np.uint8)

# img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# print(img)


import time

st = time.time()

N = 1000000
for i in range(N):
    arr1 = np.random.rand(3)
    arr2 = np.random.rand(3)

    # result = (arr1 @ arr2)[0]
    result = np.sum( arr1 * arr2 )

ed = time.time()

print(f'duration: {ed-st} sec')
