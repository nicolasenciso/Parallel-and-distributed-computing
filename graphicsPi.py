import numpy as np
import matplotlib.pyplot as plt

#time user
Y = [7.066,7.108,9.176,9.335,9.322]
X = [1,2,4,8,16]
plt.plot(X,Y,color='red')
plt.xlabel(' Number of threads')
plt.ylabel(' User Time Response')
plt.tight_layout()
plt.title(' Performance POSIX threads Pi calculus')
plt.grid()
plt.show()

#time real
Y = [7.083,3.57,2.516,2.486,2.424]
X = [1,2,4,8,16]
plt.plot(X,Y,color='red')
plt.xlabel(' Number of threads')
plt.ylabel(' Real time response')
plt.tight_layout()
plt.title(' Performance POSIX threads Pi calculus')
plt.grid()
plt.show()

#speed up user
"""
Y = [0,0.9940,0.7700,0.7569,0.7579]
X = [1,2,4,8,16]
plt.plot(X,Y,color='green')
plt.xlabel(' Number of threads')
plt.ylabel(' Speed up user time')
plt.tight_layout()
plt.title(' Performance POSIX threads Pi calculus')
plt.grid()
plt.show()
"""

#speed up real time
Y = [0,1.9840,2.8151,2.8525,2.9220]
X = [1,2,4,8,16]
plt.plot(X,Y,color='green')
plt.xlabel(' Number of threads')
plt.ylabel(' Speed up user time')
plt.tight_layout()
plt.title(' Performance POSIX threads Pi calculus')
plt.grid()
plt.show()



