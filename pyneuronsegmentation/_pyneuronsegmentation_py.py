import numpy as np
import pyneuronsegmentation as pyns

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
    
    pyns.find_neurons(framesN, framesIn, sizex, sizey, frameStride,
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
    
    NeuronYX = pyns.neuronConversion(NeuronN, NeuronXY,xyOrdering='yx')
    
    return NeuronYX, NeuronProperties
    

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
    
''''
EVERYTHING BELOW HERE WILL BE COMMENTED OUT
'''
    
    
    
    
    
    
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
    
    # If dz is set to int(1), I am asking for an array of integers that allows me
    # to index the volume stack directly
    if dz==1 and type(dz)==int:
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
    
def neuronConversionXYZtoXY(NeuronXYZ, NeuronNInVolume, volumeFirstFrame):
    '''
    volumeFirstFrame has to contain also the final+1 frame
    '''
    
    # the output is a list of the sets of neurons in each frame
    NeuronXY = []
    NeuronN = []
    Neuron = []
    nVolume = len(NeuronNInVolume)
    
    for n in np.arange(nVolume):
        Brain = np.rint(NeuronXYZ[n]).astype(np.int)
        for f in np.arange(volumeFirstFrame[n+1]-volumeFirstFrame[n]):
            ii = np.where(Brain[:,2]==f)
            neuronxy = (Brain[ii,1]//2)*256+Brain[ii,0]//2
            NeuronXY.append(neuronxy[0])
            Neuron.append(Brain[ii,0:2][0])
            NeuronN.append(len(Neuron[-1]))
            
    NeuronXY = [ne for neframe in NeuronXY for ne in neframe]
    return np.array(NeuronXY), np.array(NeuronN), Neuron
    
def curvatureConversion(NeuronCurvature, NeuronNInVolume):
    NeuronCurvatureVSplit = []
    k = 0
    for n in NeuronNInVolume:
        n = int(n)
        NeuronCurvatureVSplit.append(NeuronCurvature[k:k+n])
        k += n
        
    return NeuronCurvatureVSplit
    
def stabilizeZ(NeuronXYZ, NeuronCurvatureVSplit, method=""):
    '''
    Take the curvatures along z right above and below the found neuron and 
    stabilize the z position.
    '''
    NeuronXYZout = []
    if NeuronCurvatureVSplit[0].shape[1]==51:
        z = np.array([-3.,-2.,-1.,0.,1.,2.,3.])
        
        for i in np.arange(len(NeuronXYZ)):
            Brain = NeuronXYZ[i]
            Curvature = NeuronCurvatureVSplit[i]
            if method=="xyMaxCurvature":
                c1 = Curvature[:,0]
                c2 = np.max(Curvature[:,np.arange(1,6)],axis=1)
                c3 = np.max(Curvature[:,np.arange(6,19)],axis=1)
                c4 = np.max(Curvature[:,np.arange(19,32)],axis=1)
                c5 = np.max(Curvature[:,np.arange(32,45)],axis=1)
                c6 = np.max(Curvature[:,np.arange(45,50)],axis=1)
                c7 = Curvature[:,50]
                Curv = np.copy(np.array([c1,c2,c3,c4,c5,c6,c7]).T)
            else:
                Curv = Curvature[:,[0,3,12,25,38,47,50]] #look just along z
            
            Brain[:,2] += np.sum(z*Curv,axis=1)/np.sum(Curv,axis=1)
            NeuronXYZout.append(Brain)

    return NeuronXYZout
        
    
#def neuronConversionTuple(NeuronN, NeuronXY):

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
