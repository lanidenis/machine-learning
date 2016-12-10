import numpy as np
import matplotlib.pyplot as plt

def randSymMat(n):
    A = np.zeros((n,n))

    for i in range(0,n):
        for j in range(0,n):
            if j>=i:
                A[i,j]=np.random.normal()
            else:
                A[i,j]=A[j,i]
    return A

A = randSymMat(3)
lamb,P = np.linalg.eig(A)
Pinv = np.linalg.inv(P)

print lamb
print
print P
print Pinv
print
print np.round(np.dot(P,Pinv),8)
print np.round(np.dot(Pinv,np.dot(A,P)),8)

A = randSymMat(1000)
lamb,P = np.linalg.eig(A)
plt.plot(np.sort(lamb))
plt.show()


