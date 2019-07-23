import numpy as np
import matplotlib.pyplot as plt

#8
Y = [0.000017,0.000127,0.000626,0.000224]
X = [1,2,3,4]
plt.plot(X,Y,color='blue',label='8')
Y16 = [0.000055,0.000239,0.000363,0.000260]
X16 = [1,2,3,4]
plt.plot(X16,Y16,color='red',label='16')
Y32 = [0.000360,0.000502,0.000396,0.018802]
X32 = [1,2,3,4]
plt.plot(X32,Y32,color='orange',label='32')
Y64 = [0.002827,0.002158, 0.001543,0.001113]
X64 = [1,2,3,4]
plt.plot(X64,Y64,color='gray',label='64')
Y128 = [0.022271,0.014140,0.011856,0.009809]
X128 = [1,2,3,4]
plt.plot(X128,Y128,color='green',label='128')
Y256 = [0.269571,0.136555,0.097558,0.074840]
X256 = [1,2,3,4]
plt.plot(X256,Y256,color='purple',label='256')
Y512 = [3.780592,1.722172,1.164703,0.854753]
X512 = [1,2,3,4]
plt.plot(X512,Y512,color='yellow',label='512')
Y1024 = [30.453950,14.995240,11.140084,9.154221]
X1024 = [1,2,3,4]
plt.plot(X1024,Y1024,color='black',label='1024')
plt.xlabel(' Number of threads')
plt.ylabel(' Time(s)')
plt.tight_layout()
plt.title(' Performance openMP matrix multiply')
plt.grid()
plt.legend(loc='upper right', frameon=True)
plt.savefig('timesg.png')
plt.clf()

#speed up
sec=0.000017
Y8 = [sec/0.000017,sec/0.000127,sec/0.000626,sec/0.000224]
X = [1,2,3,4]
plt.plot(X,Y8,color='blue',label='8')

sec = 0.000055
Y16 = [sec/0.000055,sec/0.000239,sec/0.000363,sec/0.000260]
plt.plot(X,Y16,color='red',label='16')

sec = 0.000360
Y32 = [sec/0.000360,sec/0.000502,sec/0.000396,sec/0.018802]
plt.plot(X,Y32,color='orange',label='32')

sec = 0.002827
Y64 = [sec/0.002827,sec/0.002158,sec/ 0.001543,sec/0.001113]
plt.plot(X,Y64,color='gray',label='64')


sec = 0.022271
Y128 = [sec/0.022271,sec/0.014140,sec/0.011856,sec/0.009809]
plt.plot(X128,Y128,color='green',label='128')

sec = 0.269571
Y256 = [sec/0.269571,sec/0.136555,sec/0.097558,sec/0.074840]
plt.plot(X256,Y256,color='purple',label='256')

sec = 3.780592
Y512 = [sec/3.780592,sec/1.722172,sec/1.164703,sec/0.854753]
plt.plot(X512,Y512,color='yellow',label='512')

sec = 30.453950
Y1024 = [sec/30.453950,sec/14.995240,sec/11.140084,sec/9.154221]
X1024 = [1,2,3,4]
plt.plot(X1024,Y1024,color='black',label='1024')
plt.xlabel(' Number of threads')
plt.ylabel(' Speed up')
plt.tight_layout()
plt.title(' Speed up openMP matrix multiply')
plt.grid()
plt.legend(loc='upper right', frameon=True)
plt.savefig('speedup.png')
plt.clf()



