#import numpy as np
#import pyns

#N = 10
#A = np.zeros(N)

#pyns.test(N,A)

import numpy as np
#import pumpprobe as pp
import matplotlib.pyplot as plt
import pyns
import struct
import time

folder = "/home/francesco/tmp/"
folder = "/tigress/LEIFER/francesco/bleaching/20190311/pumpprobe_20190311_145010/"

N = 1

framesN = (np.uint32)(100)
sizex = (np.uint32)(512)
sizey = (np.uint32)(512)
frameStride = (np.uint32)(2)
maxFramesInVolume = 100
maxNeuronN = 100000

sizex2 = sizex // 2
sizey2 = sizey // 2
sizexy2 = sizex2*sizey2

volumeN = (np.int32)(1)
volumeFirstFrame = np.array([0, 24]).astype(np.uint32)

ArrA    = np.zeros(sizexy2, dtype=np.uint16)
ArrB    = np.zeros(sizexy2, dtype=np.float32)
ArrBX   = np.zeros(sizexy2, dtype=np.float32)
ArrBY   = np.zeros(sizexy2, dtype=np.float32)
ArrBth  = np.zeros(sizexy2, dtype=np.float32)
ArrBdil = np.zeros(sizexy2, dtype=np.float32) 

NeuronXY = np.zeros(maxNeuronN, dtype=np.uint32)
NeuronN  = np.zeros(framesN, dtype=np.uint32)


f = open(folder+'sCMOS_Frames_U16_1024x512.dat','br')
f.seek(1024*512*50)
framesIn = np.zeros((framesN,1024,512),dtype=np.uint16)
#framesIn = np.copy(np.fromfile(f, dtype=np.uint16, count=1024*512*1).reshape((512,1024)).T[0:512])
for k in np.arange(framesN):
    framesIn[k] = np.copy(np.fromfile(f, dtype=np.uint16, count=1024*512*1).reshape((512,1024)).T)//10
f.close()

#X = np.linspace(0,1,512)
#Z = (100*np.exp(-(X-0.1)**2/0.001)).astype(np.uint16)
#framesIn = np.outer(Z,Z)

t0 = time.time()
pyns.find_neurons_frames_sequence(framesN, framesIn, sizex, sizey, frameStride,
                    ArrA, ArrB, ArrBX, ArrBY, ArrBth, ArrBdil,
                    NeuronXY, NeuronN)
print((time.time()-t0)/framesN)

#print(NeuronN)
NeuronNoffset = int(np.sum(NeuronN[0:4]))
for j in np.arange(NeuronN[4]):
    xy = NeuronXY[j+NeuronNoffset]
    #print(xy)
    x = (xy//256)
    y = xy - x*256
    #x *=2
    #y *=2
    #print(x)
    #print(y)
    plt.plot(x,y,'o',markersize=3,c='r')
plt.imshow(ArrBdil.reshape((256,256)).T)
plt.show()
