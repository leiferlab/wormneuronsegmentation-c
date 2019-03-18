import numpy as np
#import pumpprobe as pp
import matplotlib.pyplot as plt
import pyneuronsegmentation as pyns
import struct
import time

folder = "/home/francesco/tmp/"
folder = "/tigress/LEIFER/francesco/bleaching/20190311/pumpprobe_20190311_145010/"

framesN = (np.uint32)(100)
sizex = (np.uint32)(512)
sizey = (np.uint32)(512)
frameStride = (np.uint32)(2)
maxFramesInVolume = 100
maxNeuronN = 100000



volumeN = (np.int32)(1)
volumeFirstFrame = np.array([0, 24]).astype(np.uint32)
'''
sizex2 = sizex // 2
sizey2 = sizey // 2
sizexy2 = sizex2*sizey2

ArrA    = np.zeros(sizexy2, dtype=np.uint16)
ArrB    = np.zeros(sizexy2, dtype=np.float32)
ArrBX   = np.zeros(sizexy2, dtype=np.float32)
ArrBY   = np.zeros(sizexy2, dtype=np.float32)
ArrBth  = np.zeros(sizexy2, dtype=np.float32)
ArrBdil = np.zeros(sizexy2, dtype=np.float32) 

NeuronXY = np.zeros(maxNeuronN, dtype=np.uint32)
NeuronN  = np.zeros(framesN, dtype=np.uint32)
'''
ArrA, ArrB, ArrBX, ArrBY, ArrBth, ArrBdil, NeuronXY, NeuronN = pyns.initVariables(framesN,sizex,sizey)


f = open(folder+'sCMOS_Frames_U16_1024x512.dat','br')
f.seek(1024*512*50)
framesIn = np.zeros((framesN,1024,512),dtype=np.uint16)
#framesIn = np.copy(np.fromfile(f, dtype=np.uint16, count=1024*512*1).reshape((512,1024)).T[0:512])
for k in np.arange(framesN):
    framesIn[k] = np.copy(np.fromfile(f, dtype=np.uint16, count=1024*512*1).reshape((512,1024)).T)
f.close()

t0 = time.time()
pyns.find_neurons_frames_sequence(framesN, framesIn, sizex, sizey, frameStride,
                    ArrA, ArrB, ArrBX, ArrBY, ArrBth, ArrBdil,
                    NeuronXY, NeuronN)
print((time.time()-t0)/framesN)

NeuronNoffset = int(np.sum(NeuronN[0:4]))
NeuronXY = NeuronXY[NeuronNoffset:NeuronNoffset+NeuronN[4]]
NeuronX = NeuronXY//256
NeuronY = NeuronXY - NeuronX*256

plt.plot(NeuronX,NeuronY,'o',markersize=3,c='r')
plt.imshow(ArrA.reshape((256,256)).T)
plt.show()
