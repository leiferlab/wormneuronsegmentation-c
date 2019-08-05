import numpy as np
import wormneuronsegmentation as wormns

def get_curvatureBoxProperties():
    boxIndices = [np.arange(1),np.arange(1,6),np.arange(6,19),np.arange(19,32),np.arange(32,45),np.arange(45,50),np.arange(50,51)]
    nPlane = 7
    
    return {'boxIndices': boxIndices, 'nPlane': nPlane}

def _findNeurons(framesIn, channelsN, volumeN, volumeFirstFrame, 
    threshold=0.25, blur=0.65, checkPlanesN=5, xydiameter=3,
    maxNeuronN=1000000, maxFramesInVolume=100, extractCurvatureBoxSize=51):
    
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
    NeuronCurvatureAll = np.zeros(maxNeuronN*extractCurvatureBoxSize, 
                                  dtype=np.float32)
    
    wormns.find_neurons(framesN, framesIn, sizex, sizey, frameStride,
                    volumeN, volumeFirstFrame,
                    ArrA, ArrBB, ArrBX, ArrBY, ArrBth, ArrBdil,
                    NeuronXYCandidatesVolume, NeuronNCandidatesVolume,
                    NeuronXYAll, NeuronNAll,
                    NeuronCurvatureAll,
                    (np.float32)(threshold), (np.float64)(blur), 
                    (np.uint32)(checkPlanesN), (np.uint32)(xydiameter),
                    (np.uint32)(extractCurvatureBoxSize))
    
    diagnostics = {"ArrA": ArrA, "ArrBB": ArrBB, "ArrBX": ArrBX, "ArrBY": ArrBY,
            "ArrBth": ArrBth, "ArrBdil": ArrBdil,
            "NeuronXYCandidatesVolume": NeuronXYCandidatesVolume,
            "NeuronNCandidatesVolume": NeuronNCandidatesVolume}
    
    NeuronNAll = NeuronNAll[0:framesN]
    NeuronTot = np.sum(NeuronNAll)
    NeuronXYAll = NeuronXYAll[0:NeuronTot]
    NeuronCurvature = NeuronCurvatureAll[0:(int)(NeuronTot*extractCurvatureBoxSize)]
    
    np.clip(NeuronCurvature,0,None,NeuronCurvature)
    NeuronCurvature = NeuronCurvature.reshape((NeuronTot,extractCurvatureBoxSize))
                    
    return NeuronNAll, NeuronXYAll, NeuronCurvature, diagnostics
    
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
    
def findNeurons(framesIn, channelsN, volumeN, volumeFirstFrame, 
    threshold=0.25, blur=0.65, checkPlanesN=5, xydiameter=3,
    maxNeuronN=1000000, maxFramesInVolume=100, extractCurvatureBoxSize=51):
    
    NeuronN, NeuronXY, NeuronCurvature, diagnostics = \
        _findNeurons(framesIn, channelsN, volumeN, volumeFirstFrame, 
            threshold,blur,checkPlanesN,xydiameter,maxNeuronN,maxFramesInVolume,
            extractCurvatureBoxSize)
            
    curvatureBoxProperties = get_curvatureBoxProperties()
    curvatureboxIndices = curvatureBoxProperties['boxIndices']
    curvatureboxNPlanes = curvatureBoxProperties['nPlane']
    
    NeuronProperties = {'curvature': NeuronCurvature, 
                        'boxNPlane': curvatureboxNPlanes, 
                        'boxIndices': curvatureboxIndices}
    
    NeuronYX = wormns.neuronConversion(NeuronN, NeuronXY,xyOrdering='yx')
    
    return NeuronYX, NeuronProperties
    
    
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
            wormns.initVariables(framesN,sizex,sizey,maxNeuronN)
    
    wormns.find_neurons_frames_sequence(
                    framesN, framesIn, sizex, sizey, frameStride,
                    ArrA, ArrB, ArrBX, ArrBY, ArrBth, ArrBdil,
                    NeuronXY, NeuronN,0.25,0.65)
                    
    return NeuronN, NeuronXY
