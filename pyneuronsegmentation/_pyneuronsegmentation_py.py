import numpy as np
import pyneuronsegmentation as pyns

def findNeuronsFramesSequence(framesIn, maxNeuronN=100000):
    '''sh = framesIn.shape
    if len(sh) == 2:
        framesN = 1
        framesStride = 1
        sizex = sh[-2]
        sizey = sh[-1]//2
    if len(sh) == 3:
        framesN = sh[0]
        framesStride = 1
        sizex = sh[-2]
        sizey = sh[-1]//2
    if len(sh) == 4:
        framesN = sh[0]
        framesStride = sh[1]
        sizex = sh[-2]
        sizey = sh[-1]//framesStride
    print(sh)'''
    framesN=framesIn.shape[0]
    framesStride=2
    sizex=512
    sizey=512
    
    framesN = (np.uint32)(framesN)
    sizex = (np.uint32)(sizex)
    sizey = (np.uint32)(sizey)
    frameStride = (np.uint32)(framesStride)
    
    ArrA, ArrB, ArrBX, ArrBY, ArrBth, ArrBdil, NeuronXY, NeuronN = \
            pyns.initVariables(framesN,sizex,sizey,maxNeuronN)
    
    pyns.find_neurons_frames_sequence(
                    framesN, framesIn, sizex, sizey, frameStride,
                    ArrA, ArrB, ArrBX, ArrBY, ArrBth, ArrBdil,
                    NeuronXY, NeuronN)
                    
    return NeuronN, NeuronXY

def initVariables(framesN,sizex,sizey,maxNeuronN=100000):
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
    
    return ArrA, ArrB, ArrBX, ArrBY, ArrBth, ArrBdil, NeuronXY, NeuronN
