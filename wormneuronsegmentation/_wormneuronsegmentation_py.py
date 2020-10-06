import numpy as np
import wormneuronsegmentation as wormns
import pkg_resources

def get_curvatureBoxProperties():
    boxIndices = [np.arange(1),np.arange(1,6),np.arange(6,19),np.arange(19,32),np.arange(32,45),np.arange(45,50),np.arange(50,51)]
    nPlane = 7
    
    return {'boxIndices': boxIndices, 'nPlane': nPlane}

def _findNeurons(framesIn, frame0, channelsN, volumeN, volumeFirstFrame, 
    threshold=0.25, blur=0.65, checkPlanesN=5, xydiameter=3,
    maxNeuronN=10000000, maxFramesInVolume=100, extractCurvatureBoxSize=51,
    candidateCheck=True,returnAll=False):
    '''
    Finds neurons in a sequence of 3D images.
    
    Parameters
    ----------
    framesIn: numpy array
        framesIn[i,ch,y,x] images. Note: Must be contiguous and row-major.
    channel: integer
        channel where to look for the neurons.
    channelsN: integer
        Number of channels present.
    volumeN: integer
        Number of volumes in which the sequence of images is to be split.
    volumeFirstFrame: numpy array of integers
        First frames of each volume. With M volumes, it must include also the
        first frame of the M+1 volume as last element (or 1+last frame of last
        volume).
    threshold: float, optional
        Threshold for noise removal. Default: 0.25
    blur: float, optional
        Gaussian blur. Default: 0.65
    checkPlanesN: integer, optional
        Number of planes determining the neighborhood along z in which to look 
        for local maxima. Default: 5
    xydiameter: integer, optional
        Number of pixel determining the neighborhood in xy in which to look for
        local maxima. Default: 3
    maxNeuronN: integer, optional
        Maximum number of neurons to expect. Used to preallocate the array in 
        which the neuron coordinates are stored. Default: 1000000
    maxFramesInVolume: integer, optional
        Maximum number of frames to expect in each volume. Used to preallocate
        arrays, as above. Default: 100
    extractCurvatureBoxSize: integer, optional
        Related to the size of the neighborhood of each neuron of which to store
        the local curvature. Right now it is hard coded, so don't use this
        parameter.
    candidateCheck: bool, optional
        If true, candidate neurons will be compared across planes.
        
    Returns
    -------
    NeuronN: numpy array
        NeuronN[i] Number of neurons found in frame i.
    NeuronXY: numpy array
        NeuronXY[j] 1-Dimensional representation of the x and y coordinates of
        neuron j (j is counted absolutely)
    NeuronCurvature: numpy array
        Local curvature around neuron j. Extraction hard coded for the time 
        being.
    diagnostics: dictionary
        Contains arrays for debugging.
    '''
    
    framesN = (np.uint32)(framesIn.shape[0])
    if len(framesIn.shape)==3:
        sizex = (np.uint32)(framesIn.shape[1])//channelsN
        sizey = (np.uint32)(framesIn.shape[2])
    elif len(framesIn.shape)==4:
        channelsN = (np.uint32)(framesIn.shape[1])
        sizex = (np.uint32)(framesIn.shape[2])
        sizey = (np.uint32)(framesIn.shape[3])
    
    frame0 = (np.int32)(frame0)
    frameStride = (np.int32)(channelsN)
    
    sizex2 = sizex // 2
    sizey2 = sizey // 2
    sizexy2 = sizex2*sizey2
    
    volumeN = (np.int32)(volumeN)
    volumeFirstFrame = np.array(volumeFirstFrame).astype(np.uint32)
    
    candidateCheck_i = int(candidateCheck)
    
    ArrA    = np.zeros(sizexy2, dtype=np.uint16)
    ArrBB   = np.zeros(sizexy2*maxFramesInVolume, dtype=np.float32)
    ArrBX   = np.zeros(sizexy2, dtype=np.float32)
    ArrBY   = np.zeros(sizexy2, dtype=np.float32)
    ArrBth  = np.zeros(sizexy2, dtype=np.float32)
    ArrBdil = np.zeros(sizexy2, dtype=np.float32) 
    
    NeuronXYCandidatesVolume = np.zeros(maxNeuronN, dtype=np.uint32)
    NeuronNCandidatesVolume = np.zeros(maxFramesInVolume, dtype=np.uint32)
    NeuronXYAll = np.zeros(maxNeuronN, dtype=np.uint32)+1
    NeuronNAll  = np.zeros(framesN, dtype=np.uint32)+1
    NeuronCurvatureAll = np.zeros(maxNeuronN*extractCurvatureBoxSize, 
                                  dtype=np.float32)
    
    wormns.find_neurons(framesN, framesIn, sizex, sizey, frame0, frameStride,
                    volumeN, volumeFirstFrame,
                    ArrA, ArrBB, ArrBX, ArrBY, ArrBth, ArrBdil,
                    NeuronXYCandidatesVolume, NeuronNCandidatesVolume,
                    NeuronXYAll, NeuronNAll,
                    NeuronCurvatureAll,
                    (np.float32)(threshold), (np.float64)(blur), 
                    (np.uint32)(checkPlanesN), (np.uint32)(xydiameter),
                    (np.uint32)(extractCurvatureBoxSize),
                    candidateCheck_i)
    
    diagnostics = {"ArrA": ArrA, "ArrBB": ArrBB, "ArrBX": ArrBX, "ArrBY": ArrBY,
                "ArrBth": ArrBth, "ArrBdil": ArrBdil,
                "NeuronXYCandidatesVolume": NeuronXYCandidatesVolume,
                "NeuronNCandidatesVolume": NeuronNCandidatesVolume}
    
    NeuronNAll = NeuronNAll[0:framesN]
    NeuronTot = np.sum(NeuronNAll)
    NeuronXYAll = NeuronXYAll[0:NeuronTot]
    # Exctract the relevant elements of NeuronCurvature. Also, change its sign
    # so that it is the actual curvature (i.e. peak = min curvature), and not 
    # the flipped version that I use in the segmentation code.
    NeuronCurvature = -1.0*NeuronCurvatureAll[0:(int)(NeuronTot*extractCurvatureBoxSize)]
    
    #np.clip(NeuronCurvature,0,None,NeuronCurvature)
    NeuronCurvature = NeuronCurvature.reshape((NeuronTot,extractCurvatureBoxSize))
    
    return NeuronNAll, NeuronXYAll, NeuronCurvature, diagnostics
    
def neuronConversion(NeuronN, NeuronXY, framesShape=(512,512), xyOrdering='xy', flattenFrames=False):
    '''Converts the results of _findNeurons to a nicer structure.
    
    Parameters
    ----------
    NeuronN: numpy array
        NeuronN[i] is the number of neurons in frame i.
    NeuronXY: numpy array
        NeuronXY[j] 1-Dimensional representation of the x and y coordinates of
        neuron j (j is counted absolutely)
    xyOredering: string, optional
        Desired output ordering of the coordinates. xy for plotting order, yx
        for indexing order. Default: xy
    flattenFrames: boolean, optional
        If True, the frames are flattened, returning the information on the 
        neurons but forgetting about their belonging to frames.
        
    Returns
    -------
    Neuron: list of numpy array
        Neuron[i][j] 2-Dimensional representation of the y and x coordinates 
        of neuron j in frame i.
    
    '''
    rowLenght = framesShape[-1]
    rowLenght2 = rowLenght//2
    
    framesN = NeuronN.shape[0]
    NeuronY = NeuronXY//rowLenght2 
    NeuronX = (NeuronXY - NeuronY*rowLenght2)
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
    
def findNeurons(framesIn, channel=0, channelsN=2, volumeN=1, 
    volumeFirstFrame=None, rectype="3d",
    threshold=0.25, blur=0.65, checkPlanesN=5, xydiameter=3,
    maxNeuronN=10000000, maxFramesInVolume=100, extractCurvatureBoxSize=51,
    candidateCheck=True, returnDiagnostics=False):
    '''
    Finds neurons in a sequence of 3D images.
    
    Parameters
    ----------
    framesIn: numpy array
        framesIn[i,ch,y,x] images. Note: Must be contiguous and row-major.
    channel: integer, optional
        channel where to look for the neurons.
    channelsN: integer, optional (needed for volumetric images)
        Number of channels present.
    volumeN: integer, optional (needed for volumetric images)
        Number of volumes in which the sequence of images is to be split.
    volumeFirstFrame: numpy array of integers, optional (needed for volumetric images)
        First frames of each volume. With M volumes, it must include also the
        first frame of the M+1 volume as last element (or 1+last frame of last
        volume).
    recType: string, optional
        3d or 2d, for volumetric or in-plane recordings. Default: 3d
    threshold: float, optional
        Threshold for noise removal. Default: 0.25
    blur: float, optional
        Gaussian blur. Default: 0.65
    checkPlanesN: integer, optional
        Number of planes determining the neighborhood along z in which to look 
        for local maxima. Default: 5
    xydiameter: integer, optional
        Number of pixel determining the neighborhood in xy in which to look for
        local maxima. Default: 3
    maxNeuronN: integer, optional
        Maximum number of neurons to expect. Used to preallocate the array in 
        which the neuron coordinates are stored. Default: 1000000
    maxFramesInVolume: integer, optional
        Maximum number of frames to expect in each volume. Used to preallocate
        arrays, as above. Default: 100
    extractCurvatureBoxSize: integer, optional
        Related to the size of the neighborhood of each neuron of which to store
        the local curvature. Right now it is hard coded, so don't use this
        parameter.
        
    Returns
    -------
    NeuronYX: list of numpy array
        NeuronYX[i][j] 2-Dimensional representation of the y and x coordinates 
        of neuron j in frame i.
    NeuronProperites: dictionary
        Contains data about the local curvature around each neuron. Only
        implemented for volumetric recording. Note: The curvature has its sign
        flipped.
    '''
    
    if rectype=="3d":
        NeuronN, NeuronXY, NeuronCurvature, diagnostics = \
            _findNeurons(framesIn, channel, channelsN, volumeN, volumeFirstFrame, 
                threshold,blur,checkPlanesN,xydiameter,maxNeuronN,
                maxFramesInVolume, extractCurvatureBoxSize, candidateCheck)
                
        curvatureBoxProperties = get_curvatureBoxProperties()
        curvatureboxIndices = curvatureBoxProperties['boxIndices']
        curvatureboxNPlanes = curvatureBoxProperties['nPlane']
        
        curvatureboxIndicesX = [np.array([10,23,36]), np.array([2,7,11,15,29,24,28,33,37,41,46]), 
            np.array([0,1,3,5,6,8,12,16,18,19,21,25,29,31,32,34,38,42,44,45,47,49,50]),
            np.array([4,9,13,17,22,26,30,35,39,43,48]), np.array([14,27,40])]
        curvatureboxIndicesY = [np.array([6,19,32]), np.array([1,7,8,9,29,21,22,33,34,35,45]),
            np.array([0,2,3,4,10,11,12,13,14,23,24,25,26,27,36,37,38,39,40,46,47,48,50]),
            np.array([5,15,16,17,28,29,30,41,42,43,49]),np.array([18,31,44])]
        
        NeuronProperties = {'curvature': NeuronCurvature, 
                            'boxNPlane': curvatureboxNPlanes, 
                            'boxIndices': curvatureboxIndices,
                            'boxIndicesX': curvatureboxIndicesX,
                            'boxIndicesY': curvatureboxIndicesY}
                            
    elif rectype=="2d":
        extractCurvatureBoxSize=13
        NeuronN, NeuronXY, NeuronCurvature = wormns.findNeuronsFramesSequence(
                            framesIn,
                            threshold=threshold,blur=blur,
                            maxNeuronN=maxNeuronN,
                            extractCurvatureBoxSize=extractCurvatureBoxSize)
        NeuronProperties = {'curvature': NeuronCurvature,
                            'boxNPlane': 1,
                            'boxIndices': [np.arange(13)],
                            'boxIndicesX': [np.array([4]),np.array([1,5,9]),np.array([0,2,6,10,12]),np.array([3,7,11]),np.array([8])],
                            'boxIndicesY': [np.array([0]),np.array([1,2,3]),np.array([4,5,6,7,8]),np.array([9,10,11]),np.array([12])]
                            }
        diagnostics = {}
        
    NeuronYX = wormns.neuronConversion(NeuronN, NeuronXY, framesShape=framesIn.shape, xyOrdering='yx')
    
    # Add parameters and module version to NeuronProperties
    version = pkg_resources.get_distribution("wormneuronsegmentation").version
    segmParam = {"threshold": threshold, "blur": blur, "version": version}
    NeuronProperties['segmParam'] = segmParam
    
    if returnDiagnostics:
        return NeuronYX, NeuronProperties, diagnostics
    else:
        return NeuronYX, NeuronProperties
    
    
def initVariables(framesN,sizex,sizey,maxNeuronN=100000,extractCurvatureBoxSize=13):
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
    NeuronCurvature = np.zeros(maxNeuronN*extractCurvatureBoxSize, dtype=np.float32)
    
    return ArrA, ArrB, ArrBX, ArrBY, ArrBth, ArrBdil, NeuronXY, NeuronN, \
            NeuronCurvature


def findNeuronsFramesSequence(framesIn, threshold=0.25, blur=0.65, 
                                maxNeuronN=100000, extractCurvatureBoxSize=13):
    '''
    Finds neurons in a sequence of 2D images (no comparisons across frames done)
    
    Parameters
    ----------
    framesIn: numpy array
        framesIn[i,y,x] images. Note: Must be contiguous and row-major.
    threshold: float
        Threshold for noise removal.
    blur: float
        Gaussian blur.
    '''
    sh = framesIn.shape
    framesN=sh[0]
    framesStride=sh[1]
    sizex=sh[2]
    sizey=sh[3]
    #framesStride=2
    #sizex=512
    #sizey=512
    
    framesN = (np.uint32)(framesN)
    sizex = (np.uint32)(sizex)
    sizey = (np.uint32)(sizey)
    frameStride = (np.uint32)(framesStride)
    threshold = (np.float32)(threshold)
    blur = (np.float64)(blur)
    extractCurvatureBoxSize = (np.uint32)(extractCurvatureBoxSize)
    
    ArrA, ArrB, ArrBX, ArrBY, ArrBth, ArrBdil, NeuronXY, NeuronN, \
            NeuronCurvature = \
            wormns.initVariables(framesN,sizex,sizey,maxNeuronN,
                                    extractCurvatureBoxSize)
    
    wormns.find_neurons_frames_sequence(
                    framesN, framesIn, sizex, sizey, frameStride,
                    ArrA, ArrB, ArrBX, ArrBY, ArrBth, ArrBdil,
                    NeuronXY, NeuronN, NeuronCurvature, 
                    threshold, blur, extractCurvatureBoxSize)
    
    NeuronTot = np.sum(NeuronN)
    # Exctract the relevant elements of NeuronCurvature. Also, change its sign
    # so that it is the actual curvature (i.e. peak = min curvature), and not 
    # the flipped version that I use in the segmentation code.
    NeuronCurvature = -1.0*NeuronCurvature[0:NeuronTot*extractCurvatureBoxSize]
    NeuronCurvature = NeuronCurvature.reshape((NeuronTot,extractCurvatureBoxSize))
                    
    return NeuronN, NeuronXY, NeuronCurvature
