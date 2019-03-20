import numpy as np
#import pumpprobe as pp
import matplotlib.pyplot as plt
import pyneuronsegmentation as pyns
import struct
import time
import unmix as um
import mistofrutta as mf

#folder = "/home/francesco/tmp/"
filename = "/tigress/LEIFER/multicolor/data/20190215/multicolorworm_20190215_154313"

Image, Brightfield = um.load_image(filename,source="spinningdisk",wormType="mcw")

framesIn = np.copy(Image[:,4,:,256:256+512]).astype(np.uint16)
framesIn = np.append(framesIn,framesIn,axis=0)

framesN = (np.uint32)(framesIn.shape[0])
sizex = (np.uint32)(framesIn.shape[1])
sizey = (np.uint32)(framesIn.shape[2])
frameStride = (np.uint32)(1)
maxFramesInVolume = 100
maxNeuronN = 100000

volumeN = (np.int32)(2)
volumeFirstFrame = np.array([0,framesN//2,framesN]).astype(np.uint32)

sizex2 = sizex // 2
sizey2 = sizey // 2
sizexy2 = sizex2*sizey2

ArrA    = np.zeros(sizexy2, dtype=np.uint16)
ArrBB    = np.zeros(sizexy2*50, dtype=np.float32)
ArrBX   = np.zeros(sizexy2, dtype=np.float32)
ArrBY   = np.zeros(sizexy2, dtype=np.float32)
ArrBth  = np.zeros(sizexy2, dtype=np.float32)
ArrBdil = np.zeros(sizexy2, dtype=np.float32) 

NeuronXYCandidatesVolume = np.zeros(maxNeuronN, dtype=np.uint32)
NeuronNCandidatesVolume = np.zeros(maxFramesInVolume, dtype=np.uint32)
NeuronXYAll = np.zeros(maxNeuronN*10, dtype=np.uint32)
NeuronNAll  = np.zeros(framesN, dtype=np.uint32)

t0 = time.time()
pyns.find_neurons(framesN, framesIn, sizex, sizey, frameStride,
                    volumeN, volumeFirstFrame,
                    ArrA, ArrBB, ArrBX, ArrBY, ArrBth, ArrBdil,
                    NeuronXYCandidatesVolume, NeuronNCandidatesVolume,
                    NeuronXYAll, NeuronNAll,
                    (np.float32)(0.05), (np.float64)(0.65))
print((time.time()-t0))


NeuronTot = np.sum(NeuronNAll[0:framesN])
print(NeuronTot)
NeuronXYAll = NeuronXYAll[:NeuronTot]
NeuronX = NeuronXYAll//256 
NeuronY = (NeuronXYAll - NeuronX*256)
NeuronX *=2
NeuronY *=2

plt.imshow(np.sum(framesIn, axis=0))
plt.plot(NeuronY,NeuronX,'o',markersize=3,c='r')
#plt.imshow(ArrA.reshape((256,256)))
plt.show()

Limits = np.append(np.zeros(1),np.cumsum(NeuronNAll)).astype(int)
Overlay = []
for i in np.arange(framesN):
    start = Limits[i]
    stop = Limits[i+1]
    Overlay.append(np.array([NeuronY[start:stop],NeuronX[start:stop]]).T)
    
mf.plt.hyperstack(framesIn,cmap='viridis',Overlay=Overlay)
