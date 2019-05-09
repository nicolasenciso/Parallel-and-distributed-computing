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
    return totalDic

def graphicTime(totalDic):
    count = 1
    for dictionary in totalDic:
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
            plt.savefig('graphics/timevsthread/'+str(count)+'.png')
            plt.clf()
            count += 1

def graphicsSpeedUp(totalDic):
    count = 1
    speed = 0
    for dictionary in totalDic:
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
            plt.savefig('graphics/speedupvsthread/'+str(count)+'.png')
            plt.clf()
            count += 1

        

graphicTime(makePoints())
graphicsSpeedUp(makePoints())