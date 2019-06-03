import matplotlib.pyplot as plt

def graphicator():
    txtfile = open('times/hd-times100.txt','r')
    timesHD, kernelSizeHD = [],[]
    for line in txtfile:
        auxLine = line.split(',')
        time = auxLine[2].split('\n')
        timesHD.append(float(time[0]))
        kernelSizeHD.append(int(auxLine[0]))
    txtfile.close()
    txtfile = open('times/fhd-times100.txt','r')
    timesFHD, kernelSizeFHD = [],[]
    for line in txtfile:
        auxLine = line.split(',')
        time = auxLine[2].split('\n')
        timesFHD.append(float(time[0]))
        kernelSizeFHD.append(int(auxLine[0]))
    txtfile.close()
    txtfile = open('times/4k-times100.txt','r')
    times4k, kernelSize4k = [],[]
    for line in txtfile:
        auxLine = line.split(',')
        time = auxLine[2].split('\n')
        times4k.append(float(time[0]))
        kernelSize4k.append(int(auxLine[0]))
    txtfile.close()
    plt.plot(timesHD,kernelSizeHD,color='red',label='HD')
    plt.plot(timesFHD,kernelSizeFHD,color='blue',label='FHD')
    plt.plot(times4k,kernelSize4k,color='green',label='4k')
    plt.legend(loc='upper right', frameon=True)
    plt.xlabel(' Time ')
    plt.ylabel(' Kernel size')
    plt.tight_layout()
    plt.title(' Kernel sizes vs time to make blur effect')
    plt.grid()
    plt.savefig('graphics/performanceCombined.png')
    plt.clf()

graphicator()
