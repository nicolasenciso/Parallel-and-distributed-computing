import matplotlib.pyplot as plt

def makePoints():
    totalDic = []
    txtfile = open('times/hd-times.txt','r')
    dictionary = {}
    for i in range(3,16):
        dictionary[i] = []
    for line in txtfile:
        auxLine = line.split(',')
        time = auxLine[2].split('\n')
        time = float(time[0])
        dictionary[int(auxLine[0])].append((int(auxLine[1]),time))
    txtfile.close()
    totalDic.append(dictionary)
    txtfile = open('times/fhd-times.txt','r')
    dictionary = {}
    for i in range(3,16):
        dictionary[i] = []
    for line in txtfile:
        auxLine = line.split(',')
        time = auxLine[2].split('\n')
        time = float(time[0])
        dictionary[int(auxLine[0])].append((int(auxLine[1]),time))
    txtfile.close()
    totalDic.append(dictionary)
    txtfile = open('times/4k-times.txt','r')
    dictionary = {}
    for i in range(3,16):
        dictionary[i] = []
    for line in txtfile:
        auxLine = line.split(',')
        time = auxLine[2].split('\n')
        time = float(time[0])
        dictionary[int(auxLine[0])].append((int(auxLine[1]),time))
    txtfile.close()
    totalDic.append(dictionary)
    return totalDic

def graphicTime(totalDic):
    count = 1
    docutype = 0
    typetxt = ''
    for dictionary in totalDic:
        docutype += 1
        if docutype == 1:
            typetxt = 'hd'
        elif docutype == 2:
            typetxt = 'fhd'
        elif docutype == 3:
            typetxt = '4k'
        for i in range(3,16):
            points = dictionary[i]
            threads, time = [],[]
            for j in range(len(points)):
                threads.append(points[j][0])
                time.append(points[j][1])
            plt.plot(threads,time,color='red')
            plt.xlabel(' Number of threads ')
            plt.ylabel(' Time')
            plt.tight_layout()
            plt.title(' Time consuming vs threads-kernel size of '+str(i))
            plt.grid()
            plt.savefig('graphics/timevsthread/'+str(typetxt)+'-'+str(count)+'.png')
            plt.clf()
            count += 1

def graphicsSpeedUp(totalDic):
    count = 1
    speed = 0
    docutype = 0
    typetxt = ''
    for dictionary in totalDic:
        docutype += 1
        if docutype == 1:
            typetxt = 'hd'
        elif docutype == 2:
            typetxt = 'fhd'
        elif docutype == 3:
            typetxt = '4k'
        for i in range(3,16):
            points = dictionary[i]
            threads, time = [],[]
            for j in range(0,len(points)):
                if j  == 0:
                    speed = points[j][1]
                time.append(float(speed/points[j][1]))
                threads.append(points[j][0])
            plt.plot(threads,time,color='blue')
            plt.xlabel(' Number of threads ')
            plt.ylabel('Speed Up')
            plt.tight_layout()
            plt.title(' Speed Up vs threads-kernel size of '+str(i))
            plt.grid()
            plt.savefig('graphics/speedupvsthread/'+str(typetxt)+'-'+str(count)+'.png')
            plt.clf()
            count += 1


def graphicCombine(dictionary):
    #print(dictionary[0][15])
    kernels = {}
    for i in range(3,16):
        kernels[i] = []
    for typeImg in dictionary:
        for i in range(3,16):
            kernels[i].append(typeImg[i])
    #print(kernels[3])
    for i in range(3,16):
        current = kernels[i]
        threadsHD, threadsFHD, threads4K = [],[],[]
        timesHD, timesFHD, times4K = [],[],[]
        hd = current[0]
        fhd = current[1]
        k4 = current[2]
        for point in hd:
            threadsHD.append(point[0])
            timesHD.append(point[1])
        for point in fhd:
            threadsFHD.append(point[0])
            timesFHD.append(point[1])
        for point in k4:
            threads4K.append(point[0])
            times4K.append(point[1])
        plt.plot(threadsHD,timesHD,color='red',label='HD')
        plt.plot(threadsFHD,timesFHD,color='blue',label='FHD')
        plt.plot(threads4K,times4K,color='green',label='4k')
        plt.legend(loc='upper right', frameon=True)
        plt.xlabel(' Number of threads ')
        plt.ylabel(' Time')
        plt.tight_layout()
        plt.title(' Time consuming vs threads-kernel size of '+str(i))
        plt.grid()
        plt.savefig('graphics/timevsthread/combined-'+str(i)+'.png')
        plt.clf()
        
def speedupCombine(dictionary):
    #print(dictionary[0][15])
    kernels = {}
    for i in range(3,16):
        kernels[i] = []
    for typeImg in dictionary:
        for i in range(3,16):
            kernels[i].append(typeImg[i])
    #print(kernels[3])
    for i in range(3,16):
        current = kernels[i]
        threadsHD, threadsFHD, threads4K = [],[],[]
        timesHD, timesFHD, times4K = [],[],[]
        hd = current[0]
        fhd = current[1]
        k4 = current[2]
        speedup = 0
        count = 0
        for point in hd:
            threadsHD.append(point[0])
            if(count == 0):
                speedup = point[1]
            timesHD.append(float(speedup/point[1]))
            count+=1
        speedup = 0
        count = 0
        for point in fhd:
            threadsFHD.append(point[0])
            if(count == 0):
                speedup = point[1]
            timesFHD.append(float(speedup/point[1]))
            count+=1
        speedup = 0
        count = 0
        for point in k4:
            threads4K.append(point[0])
            if(count == 0):
                speedup = point[1]
            times4K.append(float(speedup/point[1]))
            count+=1
        plt.plot(threadsHD,timesHD,color='red',label='HD')
        plt.plot(threadsFHD,timesFHD,color='blue',label='FHD')
        plt.plot(threads4K,times4K,color='green',label='4k')
        plt.legend(loc='upper right', frameon=True)
        plt.xlabel(' Number of threads ')
        plt.ylabel(' Time')
        plt.tight_layout()
        plt.title(' Speed Up vs threads-kernel size of '+str(i))
        plt.grid()
        plt.savefig('graphics/speedupvsthread/SpeedUpcombined-'+str(i)+'.png')
        plt.clf()




graphicTime(makePoints())
graphicsSpeedUp(makePoints())
graphicCombine(makePoints())
speedupCombine(makePoints())