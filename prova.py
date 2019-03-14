#import numpy as np
#import pyns

#N = 10
#A = np.zeros(N)

#pyns.test(N,A)

import numpy as np
import pumpprobe as pp
import matplotlib.pyplot as plt
import pyns
import time
import struct

folder = "/home/francesco/tmp/"
'''
framesN = (np.uint32)(1000)
sizex = (np.uint32)(512)
sizey = (np.uint32)(512)
frameStride = (np.uint32)(1)
maxFramesInVolume = 100

sizex2 = sizex // 2
sizey2 = sizey // 2
sizexy2 = sizex2*sizey2

volumeN = (np.int32)(1)
volumeFirstFrame = np.array([0, 24]).astype(np.uint32)

ArrA    = np.zeros(sizexy2, dtype=np.uint16)
ArrBB   = np.zeros(sizexy2*maxFramesInVolume, dtype=np.float32)
ArrBX   = np.zeros(sizexy2, dtype=np.float32)
ArrBY   = np.zeros(sizexy2, dtype=np.float32)
ArrBth  = np.zeros(sizexy2, dtype=np.float32)
ArrBdil = np.zeros(sizexy2, dtype=np.float32) 
'''

f = open(folder+'sCMOS_Frames_U16_1024x512_fr.dat','br')
N = 1
t0 = time.time()
#A = f.read(1024*512*2*N)
#A = struct.unpack(str(1024*512)+'H',f.read(1024*512*2*N))
A = np.fromfile(f, dtype=np.uint16, count=1024*512*N)
#B = np.copy(np.lib.stride_tricks.as_strided(A, shape=(N,512,512), strides=(2*1024*512,2,1024*2)))
#del A
#B = np.lib.stride_tricks.as_strided(A, shape=(2,N,512,512), strides=(512*2,1024*512*2,1024*2,2))
print((time.time()-t0)/N)
f.close()


#B = A[0:512*1024].reshape(512,1024)

'''
plt.figure(1)
plt.imshow(B[0,0])
plt.figure(2)
plt.imshow(B[1,0])
plt.show()
'''

'''

pyns.find_neurons(framesN, framesIn, sizex, sizey, frameStride,
                    volumeN, volumeFirstFrame,
                    ArrA, ArrBB, ArrBX, ArrBY, ArrBth, ArrBdil,
                    NeuronXYCandidatesVolume, NeuronNCandidatesVolume,
                    NeuronXYAll, NeuronNAll)


'''
'''
with pp.data.recording(folder) as recording:
    t0 = time.time()
    recording.load(startFrame=0, stopFrame=1000)
    print(time.time()-t0)
    print("loaded")
'''
