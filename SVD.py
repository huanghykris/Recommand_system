import numpy as np
A = np.array([[5,3],[1,1]])

lam,U=np.linalg.eig(A) # 特征分解
inv = np.linalg.inv(U) # 求逆矩阵

print(A)
print('特征值:\n',lam)
print('特征向量:\n',U)
print('特征向量的逆:\n',inv)

import numpy as np
from scipy.linalg import svd
A = np.array([[1,2],[1,1],[0,0]])
p,s,q = svd(A,full_matrices=False)
print('P=',p)
print('S=',s)
print('Q=',q)

import numpy as np
from scipy.linalg import svd
from PIL import Image
import matplotlib.pyplot as plt

# 取前k个特征，对图像进行还原
def get_image_feature(s,k):
    # 对于S，只保留前K个特征值
    s_temp = np.zeros(s.shapep[0])
    s_temp[0:k] = s[0:k]
    s = s_temp * np.identity(s.shape[0])
    # 用新的s_temp,以及p,q重构A
    temp = np.dot(p,s)
    temp = np.dot(temp,q)
    plt.imshow(temp,cmap=plt.cm.gray,interpolation='nearest')
    plt.show()
    print(A-temp)