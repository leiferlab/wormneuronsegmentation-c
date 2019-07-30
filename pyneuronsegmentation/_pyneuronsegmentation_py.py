import numpy as np
import pyneuronsegmentation as pyns

def findNeurons(framesIn, channelsN, volumeN, volumeFirstFrame, 
    threshold=0.25, blur=0.65, checkPlanesN=5, xydiameter=3,
    maxNeuronN=100000, maxFramesInVolume=100):
    
    framesN = (np.uint32)(framesIn.shape[0])
    if len(framesIn.shape)==3:
        sizex = (np.uint32)(framesIn.shape[1])//channelsN
        sizey = (np.uint32)(framesIn.shape[2])
    elif len(framesIn.shape)==4:
        channelsN = (np.uint32)(framesIn.shape[1])
        sizex = (np.uint32)(framesIn.shape[2])
        sizey = (np.uint32)(framesIn.shape[3])
    frameStride = (np.int32)(channelsN)
    
    sizex2 = sizex // 2
    sizey2 = sizey // 2
    sizexy2 = sizex2*sizey2
    
    volumeN = (np.int32)(volumeN)
    volumeFirstFrame = np.array(volumeFirstFrame).astype(np.uint32)
    
    ArrA    = np.zeros(sizexy2, dtype=np.uint16)
    ArrBB    = np.zeros(sizexy2*maxFramesInVolume, dtype=np.float32)
    ArrBX   = np.zeros(sizexy2, dtype=np.float32)
    ArrBY   = np.zeros(sizexy2, dtype=np.float32)
    ArrBth  = np.zeros(sizexy2, dtype=np.float32)
    ArrBdil = np.zeros(sizexy2, dtype=np.float32) 
    
    NeuronXYCandidatesVolume = np.zeros(maxNeuronN, dtype=np.uint32)
    NeuronNCandidatesVolume = np.zeros(maxFramesInVolume, dtype=np.uint32)
    NeuronXYAll = np.zeros(maxNeuronN, dtype=np.uint32)
    NeuronNAll  = np.zeros(framesN, dtype=np.uint32)
    
    pyns.find_neurons(framesN, framesIn, sizex, sizey, frameStride,
                    volumeN, volumeFirstFrame,
                    ArrA, ArrBB, ArrBX, ArrBY, ArrBth, ArrBdil,
                    NeuronXYCandidatesVolume, NeuronNCandidatesVolume,
                    NeuronXYAll, NeuronNAll,
                    (np.float32)(threshold), (np.float64)(blur), 
                    (np.uint32)(checkPlanesN), (np.uint32)(xydiameter))
    
    diagnostics = {"ArrA": ArrA, "ArrBB": ArrBB, "ArrBX": ArrBX, "ArrBY": ArrBY,
            "ArrBth": ArrBth, "ArrBdil": ArrBdil,
            "NeuronXYCandidatesVolume": NeuronXYCandidatesVolume,
            "NeuronNCandidatesVolume": NeuronNCandidatesVolume}
    
    NeuronNAll = NeuronNAll[0:framesN]
    NeuronTot = np.sum(NeuronNAll)
    NeuronXYAll = NeuronXYAll[0:NeuronTot]
                    
    return NeuronNAll, NeuronXYAll, diagnostics
    
def neuronConversion(NeuronN, NeuronXY, xyOrdering='xy', flattenFrames=False):
    framesN = NeuronN.shape[0]
    NeuronY = NeuronXY//256 
    NeuronX = (NeuronXY - NeuronY*256)
    NeuronX *=2
    NeuronY *=2

    if flattenFrames: 
        if xyOrdering=='xy':
            return np.array([NeuronX,NeuronY]).T
        elif xyOrdering=='yx':
            return np.array([NeuronY,NeuronX]).T
            
    Limits = np.append(np.zeros(1),np.cumsum(NeuronN)).astype(int)
    Neuron = []
    for i in np.arange(framesN):
        start = Limits[i]
        stop = Limits[i+1]
        if xyOrdering=='xy':
            Neuron.append(np.array([NeuronX[start:stop],NeuronY[start:stop]]).T)
        elif xyOrdering=='yx':
            Neuron.append(np.array([NeuronY[start:stop],NeuronX[start:stop]]).T)
        
    return Neuron
    
def neuronConversionFromFlattenedFrames(NeuronN, Neuron_fl, xyOrdering='xy'):
    if xyOrdering=="xy":
        NeuronX, NeuronY = Neuron_fl.T
    elif xyOrdering=="yx":
        NeuronY, NeuronX = Neuron_fl.T
    
    framesN = NeuronN.shape[0]
    Limits = np.append(np.zeros(1),np.cumsum(NeuronN)).astype(int)
    Neuron = []
    for i in np.arange(framesN):
        start = Limits[i]
        stop = Limits[i+1]
        if xyOrdering=='xy':
            Neuron.append(np.array([NeuronX[start:stop],NeuronY[start:stop]]).T)
        elif xyOrdering=='yx':
            Neuron.append(np.array([NeuronY[start:stop],NeuronX[start:stop]]).T)
        
    return Neuron
    
def neuronConversionXYZ(NeuronN, NeuronXY, volumeFirstFrame,ZZ=[],dz=2.4,xyOrdering='xyz'):
    #ZZ[volume,frame]

    framesN = NeuronN.shape[0]
    NeuronY = NeuronXY//256 
    NeuronX = (NeuronXY - NeuronY*256)
    NeuronX *=2
    NeuronY *=2
    
    Limits = np.append(np.zeros(1),np.cumsum(NeuronN)).astype(int)
    Neuron = []
    NeuronNInVolume = []
    L = len(volumeFirstFrame)-1
    
    # If dz is set to 1, I am asking for an array of integers that allows me
    # to index the volume stack directly
    if dz==1:
        tipo=np.int
    else:
        tipo=np.float
    
    #For each volume
    for l in np.arange(L):
        firstframe = volumeFirstFrame[l]
        lastframeplusone = volumeFirstFrame[l+1]
        NeuronNInVolume.append(np.sum(NeuronN[firstframe:lastframeplusone]))
        
        NeuronInVolume = np.zeros((NeuronNInVolume[-1],3),dtype=tipo)

        # For each frame in the volume
        for i in np.arange(firstframe, lastframeplusone):
            start = Limits[i]
            stop = Limits[i+1]
            if len(ZZ)==0:
                Z = np.ones(stop-start,dtype=tipo)*(i-firstframe)*dz
            else:
                Z = np.ones(stop-start)*ZZ[l][i-firstframe]
            if xyOrdering=='xyz':
                NeuronInVolume[start-Limits[firstframe]:stop-Limits[firstframe]] = \
                                             np.array([NeuronX[start:stop],
                                                       NeuronY[start:stop],
                                                       Z]).T
            elif xyOrdering=='zyx':
                NeuronInVolume[start-Limits[firstframe]:stop-Limits[firstframe]] = \
                                             np.array([Z,
                                                       NeuronY[start:stop],
                                                       NeuronX[start:stop]]).T
        Neuron.append(NeuronInVolume)
        
    return Neuron, NeuronNInVolume
    
#def neuronConversionTuple(NeuronN, NeuronXY):
    

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
                    NeuronXY, NeuronN,0.25,0.65)
                    
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
