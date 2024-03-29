/**
    wormneuronsegmentation
    neuronsegmentation.cpp
    Finds nuclei of neurons in stack of fluorescence images.

    @author Francesco Randi
**/

#include <stdint.h>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "neuronsegmentation.hpp"
#include <iostream>
#include <queue>

void find_neurons_frames_sequence_c(uint16_t framesIn[],
    uint32_t framesN, int32_t sizex, int32_t sizey,
    int32_t framesStride, // 1 or 2 (RFP RFP RFP or RFP GFP RFP GFP)
    uint16_t ArrA[], 
    float ArrB[], float ArrBX[], float ArrBY[], 
	float ArrBth[], float ArrBdil[],
	uint32_t NeuronXY[], uint32_t NeuronN[],
	float NeuronCurvature[],
	float threshold, double blur, uint32_t dil_size, uint16_t A_thresh,
	uint32_t extractCurvatureBoxSize
	) {
	
    // Size of the arrays / images. sizex2 and sizey2 are the sizes of the 
    // resized images.
    int32_t sizex2, sizey2, sizexy;
    sizex2 = sizex / 2;
    sizey2 = sizey / 2;
    //sizexy2 = sizex2*sizey2;
    sizexy = sizex*sizey;
    
    // Declare the arrays that you don't need to be accessible from outside C++.
    int32_t sizeC = 5;
    float ArrC[5] = {-2.0, 1.0, 2.0, 1.0, -2.0};
    
    // Pointer to ImgIn inside framesIn[]
    uint16_t * ImgIn;
    
    //Pointer to specific positions in the NeuronCurvatureAll array.
	float *NeuronCurvaturePt = NeuronCurvature;
       
    // Create the cv::Mat header for all the images.
    cv::Mat A = cv::Mat(sizex2, sizey2, CV_16U, ArrA);
    cv::Mat B = cv::Mat(sizex2, sizey2, CV_32F, ArrB);
	cv::Mat BX = cv::Mat(sizex2, sizey2, CV_32F, ArrBX);
	cv::Mat BY = cv::Mat(sizex2, sizey2, CV_32F, ArrBY);
	cv::Mat Bth = cv::Mat(sizex2, sizey2, CV_32F, ArrBth);
	cv::Mat Bdil = cv::Mat(sizex2, sizey2, CV_32F, ArrBdil);
	cv::Mat C = cv::Mat(sizeC, 1, CV_32F, ArrC);
	cv::Mat OneHalf = cv::Mat(1, 1, CV_32F, cv::Scalar::all(0.5));
	cv::Mat K = cv::Mat(dil_size, dil_size, CV_32F, cv::Scalar::all(1));
	    
    double maxX, maxY, maxXInStack, maxYInStack;
    //float threshold = 0.25;//0.25
            
    // Index for the frames with respect to the beginning
    ImgIn = framesIn;
    
    maxXInStack = -1.0;
    maxYInStack = -1.0;
    int chunk_tot_count;
    int chunk_tot_thresh = 10;
        
    // Run the single frame segmentation to initialize maxX/YInStack
    segment_singleframe_pipeline(ImgIn, sizex, sizey, 
        C, sizeC, A, B, BX, BY, Bth, Bdil, K, 
        NeuronXY, NeuronN[0],
        maxX, maxY, maxXInStack, maxYInStack, 
        chunk_tot_count, chunk_tot_thresh,
        threshold, blur, A_thresh, true);
        
    maxXInStack = maxX;
    maxYInStack = maxY;
        
    // Segment each frame independently and find the neurons.
    for(uint nu=0; nu<framesN; nu++) {             
        // Run the single frame segmentation.
        segment_singleframe_pipeline(ImgIn, sizex, sizey, 
            C, sizeC, A, B, BX, BY, Bth, Bdil, K, 
            NeuronXY, NeuronN[nu],
            maxX, maxY, maxXInStack, maxYInStack, 
            chunk_tot_count, chunk_tot_thresh,
            threshold, blur, A_thresh, true);
        
        segment_extract_curvature_single_frame(ArrB,sizex2,sizey2,
                    NeuronXY, NeuronN[nu],
	                NeuronCurvaturePt, extractCurvatureBoxSize);
        
        if(framesN>0){
            /**
            Move the pointer to the next frames in framesIn/ImgIn.
            For the time being, i is just to keep track of the current frame, 
            but I'm not using it actually for anything.
            **/
            ImgIn = ImgIn + sizexy*framesStride;
            
            /**
            Move NeuronXY further along NeuronXYCAndidatesVolume by 
            the number of candidates found in this volume, and 
            NeuronNCandidates to the next element of NeuronNCandidatesVolume.
            **/
            NeuronXY = NeuronXY + NeuronN[nu];
            NeuronCurvaturePt = NeuronCurvaturePt + NeuronN[nu]*extractCurvatureBoxSize;
            
            // Update encountered maxima.
            maxXInStack = maxX;
            maxYInStack = maxY;
        }
    }
}

void find_neurons(uint16_t framesIn[],
    uint32_t framesN, int32_t sizex, int32_t sizey,
    int32_t frame0, //channel where to do the segmentation (0 RFP, 1 GFP, ...)
    int32_t framesStride, // 1 or 2 (RFP RFP RFP or RFP GFP RFP GFP)
    uint32_t volumeFirstFrame[], uint32_t volumeN,
    uint16_t ArrAA[], //TODO AA
    float ArrBB[], float ArrBX[], float ArrBY[], 
	float ArrBth[], float ArrBdil[],
	uint32_t NeuronXYCandidatesVolume[], 
	uint32_t NeuronNCandidatesVolume[],
	uint32_t NeuronXYAll[], uint32_t NeuronNAll[],
	float NeuronCurvatureAll[],
	float threshold, double blur, uint32_t dil_size, uint16_t A_thresh,
	uint32_t checkPlanesN, uint32_t xydiameter,
	uint32_t extractCurvatureBoxSize, bool candidateCheck,
	int32_t maxNeuronNPerVolume
	) {
	
	/*
	NeuronXYCandidatesVolume and NeuronNCandidatesVolume will store the
	candidate neurons for the volume being processed. The candidates that are
	selected will be stored in NeuronXYAll and NeuronNAll. (Still ordered by
	frames, but after being selected in 3D inside the volume). Therefore, the 
	length of NeuronNAll will be the number of frames.
	The pointers *Neuron--Candidates point at specific locations in
	the arrays Neuron--CandidatesVolume, and are the ones that are passed to
	segment_single_frame(), so that the sigle-frame results can be accumulated
	in the Neuron--CandidatesVolume arrays for the subsequent candidates-
	selection.
	*/
	
	uint32_t *NeuronXYCandidates;
	
	//This will be needed to correctly move the pointer in NeuronXYAll.
	uint32_t NeuronNInAllPreviousVolumes = 0;
	
	//Pointer to specific positions in the NeuronCurvatureAll array.
	float *NeuronCurvature = NeuronCurvatureAll;
	
	//This is the total size of the box out of which curvatures are extracted
	//for the "watershed".
	// passed as parameter extractCurvatureBoxSize
    
    // Size of the arrays / images. sizex2 and sizey2 are the sizes of the 
    // resized images.
    int32_t sizex2, sizey2, sizexy2, sizexy;
    sizex2 = sizex / 2;
    sizey2 = sizey / 2;
    sizexy2 = sizex2*sizey2;
    sizexy = sizex*sizey;
    
    // Declare the arrays that you don't need to be accessible from outside C++.
    int32_t sizeC = 5;
    float ArrC[5] = {-2.0, 1.0, 2.0, 1.0, -2.0};
    //int32_t sizeC = 7;
    //float ArrC[7] = {-2.0,0.0,1.2,1.6,1.2,0.0,-2.0};
    //int32_t sizeC = 13;
    //float ArrC[13] = {-2.0, -1.0, -0.1818182, 0.4545454, 0.9090909,1.1818182, 1.27272739, 1.1818182, 0.90909099, 0.4545454, -0.1818182, -1.0, -2.0};
    
    // Zero is used later in the selection of the candidate neurons.   
    uint16_t * Zero_uint16;
    Zero_uint16 = new uint16_t[sizexy2];
    for(int k=0; k<sizexy2; k++) { Zero_uint16[k] = 0; }
    
    float * Zero;
    Zero = new float[sizexy2];
    for(int k=0; k<sizexy2; k++) { Zero[k] = 0.0; }
    
    // Pointer to ImgIn inside framesIn[]
    uint16_t * ImgIn;
    
    // Pointer to ArrA inside ArrAA
    uint16_t *ArrA; //TODO AA
    
    // Pointer to ArrB inside ArrBB
    float *ArrB;
    
    // Pointers to B0, B1, B2, B3, B4 for the candidate check.
    uint16_t *A0, *A1, *A2, *A3, *A4; //TODO AA
    float *B0, *B1, *B2, *B3, *B4;
    
    // Create the cv::Mat header for all the images but B, which will need to 
    // be changed at every frame within a volume, so that they can be stored.
	cv::Mat BX = cv::Mat(sizex2, sizey2, CV_32F, ArrBX);
	cv::Mat BY = cv::Mat(sizex2, sizey2, CV_32F, ArrBY);
	cv::Mat Bth = cv::Mat(sizex2, sizey2, CV_32F, ArrBth);
	cv::Mat Bdil = cv::Mat(sizex2, sizey2, CV_32F, ArrBdil);
	cv::Mat C = cv::Mat(sizeC, 1, CV_32F, ArrC);
	cv::Mat OneHalf = cv::Mat(1, 1, CV_32F, cv::Scalar::all(0.5));
	cv::Mat K = cv::Mat(dil_size, dil_size, CV_8U, cv::Scalar::all(1));
    for(int i=0;i<dil_size;i++){
    for(int j=0;j<std::abs(i-(int)dil_size/2);j++){
        K.at<uint8_t>(i*dil_size+j) = 0;
        K.at<uint8_t>(i*(dil_size+1)-j) = 0;
    }}
	
	cv::Mat A; //TODO AA
	cv::Mat B;
       
    double maxX, maxY, maxXInStack, maxYInStack, maxXInStackOld, maxYInStackOld;
    //float threshold = 0.25;
    
    // Index for the frames with respect to the beginning
    int i = 0;
    ImgIn = framesIn + sizexy*frame0;
    
    // Point to the first allocated B frame in ArrBB. The pointer ArrB will
    // be moved as the analysis proceeds through the volume.
    ArrB = ArrBB;
    ArrA = ArrAA; //TODO AA
    
    // Same as above
    NeuronXYCandidates = NeuronXYCandidatesVolume;
    
    // Initialize max_InStack with the first volume
    maxXInStack = -1.0;
    maxYInStack = -1.0;
    
    // Initialize chunk_tot_count and chunk_tot_thresh
    int chunk_tot_count;
    int chunk_tot_max_in_stack=0;
    int chunk_tot_thresh=10;
    
    int framesInVolume = volumeFirstFrame[1] - volumeFirstFrame[0];
    
    // Segment each frame nu independently and find the candidate neurons.
    for(int nu=0; nu<framesInVolume; nu++) {
        // Create the Mat header for the pointer ArrB, that changes at every
        // iteration.
        A = cv::Mat(sizex2, sizey2, CV_16U, ArrA); //TODO AA
        B = cv::Mat(sizex2, sizey2, CV_32F, ArrB);
        
        // With the adjusted pointers and Mat headers, run the single frame
        // segmentation.
        segment_singleframe_pipeline(ImgIn, sizex, sizey, 
            C, sizeC, A, B, BX, BY, Bth, Bdil, K, 
            NeuronXYCandidates, NeuronNCandidatesVolume[nu],
            maxX, maxY, maxXInStack, maxYInStack, 
            chunk_tot_count, chunk_tot_thresh,
            threshold, blur, A_thresh, true);
        
        // Move pointer to frames to the next one
        ImgIn = ImgIn + sizexy*framesStride;
                
        // Update encountered maxima.
        maxXInStack = (maxXInStack<maxX)?maxX:maxXInStack;
        maxYInStack = (maxYInStack<maxY)?maxY:maxYInStack;
        if (chunk_tot_max_in_stack<chunk_tot_count) {
            chunk_tot_max_in_stack = chunk_tot_count;
        }
    }
    
    maxXInStackOld = maxXInStack;
    maxYInStackOld = maxYInStack;
    chunk_tot_thresh = chunk_tot_max_in_stack;
    
    // Bring pointer to frames to the beginning for the actual scan
    ImgIn = framesIn;
   
    // For each volume mu
    for(uint mu=0; mu<volumeN; mu++) {
        // Point to the first allocated B frame in ArrBB. The pointer ArrB will
        // be moved as the analysis proceeds through the volume.
        ArrA = ArrAA; //TODO AA
        ArrB = ArrBB;
        
        // Same as above
        NeuronXYCandidates = NeuronXYCandidatesVolume;
        //NeuronXYCandidates = NeuronXYAll;
        
        // For volume mu, the segmentation function takes the max_InStack from
        // volume mu-1 (or the initialization above for mu=0), max_InStackOld,
        // while it finds also the max_InStack of volume mu. To do the latter,
        // reinitialize max_InStack at the beginning of every volume.
        maxXInStack = -1.0;
        maxYInStack = -1.0;
        maxX = 0.0;
        maxY = 0.0;
        chunk_tot_max_in_stack=0;
        
        int framesInVolume = volumeFirstFrame[mu+1] - volumeFirstFrame[mu];
        
        // Segment each frame nu independently and find the candidate neurons.
        for(int nu=0; nu<framesInVolume; nu++) {
            
            // Create the Mat header for the pointer ArrB, that changes at every
            // iteration.
            A = cv::Mat(sizex2, sizey2, CV_16U, ArrA); //TODO AA
            B = cv::Mat(sizex2, sizey2, CV_32F, ArrB);
            
            // With the adjusted pointers and Mat headers, run the single frame
            // segmentation.
            segment_singleframe_pipeline(
                ImgIn, sizex, sizey, 
                C, sizeC, A, B, BX, BY, Bth, Bdil, K, 
                NeuronXYCandidates, NeuronNCandidatesVolume[nu],
                maxX, maxY, maxXInStackOld, maxYInStackOld, 
                chunk_tot_count, chunk_tot_thresh,
                threshold, blur, A_thresh, true);
            /**
            Move the pointers to the next frames in framesIn/ImgIn and 
            ArrBB/ArrB. While ArrB is resent to point to the beginning of ArrBB
            at each new volume, ImgIn just keeps going. For the time being, i
            is just to keep track of the current frame, but I'm not using it
            actually for anything.
            **/
            i++;
            ImgIn = ImgIn + sizexy*framesStride;
            ArrA = ArrA + sizexy2; //TODO AA
            ArrB = ArrB + sizexy2;
            
            /**
            Move NeuronXYCandidates further along NeuronXYCAndidatesVolume by 
            the number of candidates found in this volume.
            **/
            NeuronXYCandidates += NeuronNCandidatesVolume[nu];
            //NeuronNAll[nu] = NeuronNCandidatesVolume[nu];
            
            // Update encountered maxima.
            maxXInStack = (maxXInStack<maxX)?maxX:maxXInStack;
            maxYInStack = (maxYInStack<maxY)?maxY:maxYInStack;
            if (chunk_tot_max_in_stack<chunk_tot_count) {
                chunk_tot_max_in_stack = chunk_tot_count;
            }

        }
        
        // FIXME Well, this is practically hard-coding the tagRFP bleaching, so it has to be fixed..
        // Any other thing (like using the the max_InStack from the previous volume
        // produces a bleaching effect... I cannot understand why. And also I don't recall seeing this
        // in my recordings (on the new instrument).
        //maxXInStack0 = maxXInStack0*0.9995;
        //maxYInStack0 = maxYInStack0*0.9995;
        //maxXInStackOld = maxXInStack;
        //maxYInStackOld = maxYInStack;
        if(maxXInStack<3.*maxXInStackOld && maxYInStack<3.*maxYInStackOld &&
            maxXInStack != 0. && maxYInStack != 0.0
            ){
            maxXInStackOld = 0.75*maxXInStackOld+0.25*maxXInStack; //0.75  + 0.25
            maxYInStackOld = 0.75*maxYInStackOld+0.25*maxYInStack;
        }
        chunk_tot_thresh = chunk_tot_max_in_stack;
        
        /**
        After the candidate neurons are found, select them using all the B 
        stored in ArrBB. 
        When iterating over nu = 0, 1, 2, framesN-1, framesN-2, framesN-3,
        pass the array Zero accordingly.
        **/
        
        int nu;
        uint32_t *NeuronXYin, *NeuronXYout;
        uint32_t NeuronNin;
        
        if(candidateCheck){
        if(checkPlanesN==5){
            // with B2 at nu=0 (B0=B1=Zero)
            nu = 0;
            B0 = Zero;
            B1 = Zero;
            B2 = ArrBB;
            B3 = ArrBB + sizexy2;
            B4 = ArrBB + 2*sizexy2;
            NeuronXYin = NeuronXYCandidatesVolume;
            NeuronNin = NeuronNCandidatesVolume[0];
            NeuronXYout = NeuronXYAll + NeuronNInAllPreviousVolumes;
                    
            segment_check2dcandidates_5planes(B0, B1, B2, B3, B4, 
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
                    maxNeuronNPerVolume);

            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            
            // with B2 at nu=1 (B0=Zero)
            nu = 1;
            B0 = Zero;
            B1 = ArrBB;
            B2 = ArrBB + sizexy2;
            B3 = ArrBB + 2*sizexy2;
            B4 = ArrBB + 3*sizexy2;
            NeuronXYin += NeuronNin;
            NeuronNin = NeuronNCandidatesVolume[1];
            NeuronXYout += NeuronNAll[volumeFirstFrame[mu]+nu-1];
            
            segment_check2dcandidates_5planes(B0, B1, B2, B3, B4, 
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
                    maxNeuronNPerVolume);
            
            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            
            for(int nu=2; nu<framesInVolume-2; nu++) {
                // Move the pointers to the right positions.
                
                // The actual "Hessians"
                B0 = ArrBB + (nu-2)*sizexy2;
                B1 = ArrBB + (nu-1)*sizexy2;
                B2 = ArrBB + (nu)*sizexy2;
                B3 = ArrBB + (nu+1)*sizexy2;
                B4 = ArrBB + (nu+2)*sizexy2;
                
                // The indexes and number of the neuron candidates.
                NeuronXYin += NeuronNin;
                NeuronNin = NeuronNCandidatesVolume[nu];
                NeuronXYout += NeuronNAll[volumeFirstFrame[mu]+nu-1];
                
                segment_check2dcandidates_5planes(B0, B1, B2, B3, B4, 
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
                    maxNeuronNPerVolume);
                
                NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            }
            
            // with B2 at nu=framesInVolume-2 (B4=Zero)
            int nu = framesInVolume - 2;
            B0 = ArrBB + (framesInVolume-4)*sizexy2;
            B1 = ArrBB + (framesInVolume-3)*sizexy2;
            B2 = ArrBB + (framesInVolume-2)*sizexy2;
            B3 = ArrBB + (framesInVolume-1)*sizexy2;
            B4 = Zero;
            NeuronXYin += NeuronNin;
            NeuronNin = NeuronNCandidatesVolume[nu];
            NeuronXYout += NeuronNAll[volumeFirstFrame[mu]+nu-1];
            
            segment_check2dcandidates_5planes(B0, B1, B2, B3, B4, 
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
                    maxNeuronNPerVolume);
                    
            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            
            // with B2 at nu=framesInVolume-1 (B4=B3=Zero)
            nu = framesInVolume - 1;
            B0 = ArrBB + (framesInVolume-3)*sizexy2;
            B1 = ArrBB + (framesInVolume-2)*sizexy2;
            B2 = ArrBB + (framesInVolume-1)*sizexy2;
            B3 = Zero;
            B4 = Zero;
            NeuronXYin += NeuronNin;
            NeuronNin = NeuronNCandidatesVolume[nu];
            NeuronXYout += NeuronNAll[volumeFirstFrame[mu]+nu-1];
            
            segment_check2dcandidates_5planes(B0, B1, B2, B3, B4, 
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
                    maxNeuronNPerVolume);
            
            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
        
        } else if(checkPlanesN==7) {
            // You need two additional pointers
            uint16_t *A5, *A6; //TODO AA
            float *B5, *B6;
        
            // with B3 at nu=0 (B0=B1=B2=Zero)
            nu = 0;
            
            A0 = Zero_uint16; //TODO AA
            A1 = Zero_uint16;
            A2 = Zero_uint16;
            A3 = ArrAA;
            A4 = ArrAA + sizexy2;
            A5 = ArrAA + 2*sizexy2;
            A6 = ArrAA + 3*sizexy2;
            
            B0 = Zero;
            B1 = Zero;
            B2 = Zero;
            B3 = ArrBB;
            B4 = ArrBB + sizexy2;
            B5 = ArrBB + 2*sizexy2;
            B6 = ArrBB + 3*sizexy2;
            NeuronXYin = NeuronXYCandidatesVolume;
            NeuronNin = NeuronNCandidatesVolume[0];
            NeuronXYout = NeuronXYAll + NeuronNInAllPreviousVolumes;
                    
            segment_check2dcandidates_7planes(A0, A1, A2, A3, A4, A5, A6, //TODO AA
                    B0, B1, B2, B3, B4, B5, B6,
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
                    xydiameter, maxNeuronNPerVolume);
                    
            /*segment_extract_curvature(B0,B1,B2,B3,B4,B5,B6,sizex2,sizey2,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
	                NeuronCurvature, extractCurvatureBoxSize);*/
            NeuronCurvature += NeuronNAll[volumeFirstFrame[mu]+nu]*extractCurvatureBoxSize;

            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            
            // with B3 at nu=1 (B0=B1=Zero)
            nu = 1;
            
            A0 = Zero_uint16;
            A1 = Zero_uint16;
            A2 = ArrAA;
            A3 = ArrAA + sizexy2;
            A4 = ArrAA + 2*sizexy2;
            A5 = ArrAA + 3*sizexy2;
            A6 = ArrAA + 4*sizexy2;
            
            B0 = Zero; //TODO AA
            B1 = Zero;
            B2 = ArrBB;
            B3 = ArrBB + sizexy2;
            B4 = ArrBB + 2*sizexy2;
            B5 = ArrBB + 3*sizexy2;
            B6 = ArrBB + 4*sizexy2;
            NeuronXYin += NeuronNin;
            NeuronNin = NeuronNCandidatesVolume[1];
            NeuronXYout += NeuronNAll[volumeFirstFrame[mu]+nu-1];
            
            segment_check2dcandidates_7planes(A0, A1, A2, A3, A4, A5, A6, //TODO AA
                    B0, B1, B2, B3, B4, B5, B6,
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
                    xydiameter, maxNeuronNPerVolume);
                    
            /*segment_extract_curvature(B0,B1,B2,B3,B4,B5,B6,sizex2,sizey2,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
	                NeuronCurvature, extractCurvatureBoxSize);*/
            NeuronCurvature += NeuronNAll[volumeFirstFrame[mu]+nu]*extractCurvatureBoxSize;
            
            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            
            // with B3 at nu=2 (B0=Zero)
            nu = 2;
            
            A0 = Zero_uint16; //TODO AA
            A1 = ArrAA;
            A2 = ArrAA + sizexy2;
            A3 = ArrAA + 2*sizexy2;
            A4 = ArrAA + 3*sizexy2;
            A5 = ArrAA + 4*sizexy2;
            A6 = ArrAA + 5*sizexy2;
            
            B0 = Zero;
            B1 = ArrBB;
            B2 = ArrBB + sizexy2;
            B3 = ArrBB + 2*sizexy2;
            B4 = ArrBB + 3*sizexy2;
            B5 = ArrBB + 4*sizexy2;
            B6 = ArrBB + 5*sizexy2;
            NeuronXYin += NeuronNin;
            NeuronNin = NeuronNCandidatesVolume[1];
            NeuronXYout += NeuronNAll[volumeFirstFrame[mu]+nu-1];
            
            segment_check2dcandidates_7planes(A0, A1, A2, A3, A4, A5, A6, //TODO AA
                    B0, B1, B2, B3, B4, B5, B6,
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
                    xydiameter, maxNeuronNPerVolume);
                    
            /*segment_extract_curvature(B0,B1,B2,B3,B4,B5,B6,sizex2,sizey2,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
	                NeuronCurvature, extractCurvatureBoxSize);*/
            NeuronCurvature += NeuronNAll[volumeFirstFrame[mu]+nu]*extractCurvatureBoxSize;
            
            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            
            for(int nu=3; nu<framesInVolume-3; nu++) {
                // Move the pointers to the right positions.
                
                A0 = ArrAA + (nu-3)*sizexy2; //TODO AA
                A1 = ArrAA + (nu-2)*sizexy2;
                A2 = ArrAA + (nu-1)*sizexy2;
                A3 = ArrAA + (nu)*sizexy2;
                A4 = ArrAA + (nu+1)*sizexy2;
                A5 = ArrAA + (nu+2)*sizexy2;
                A6 = ArrAA + (nu+3)*sizexy2;                
                                
                // The actual "Hessians"
                B0 = ArrBB + (nu-3)*sizexy2;
                B1 = ArrBB + (nu-2)*sizexy2;
                B2 = ArrBB + (nu-1)*sizexy2;
                B3 = ArrBB + (nu)*sizexy2;
                B4 = ArrBB + (nu+1)*sizexy2;
                B5 = ArrBB + (nu+2)*sizexy2;
                B6 = ArrBB + (nu+3)*sizexy2;
                
                // The indexes and number of the neuron candidates.
                NeuronXYin += NeuronNin;
                NeuronNin = NeuronNCandidatesVolume[nu];
                NeuronXYout += NeuronNAll[volumeFirstFrame[mu]+nu-1];
                
                segment_check2dcandidates_7planes(A0, A1, A2, A3, A4, A5, A6, //TODO AA
                    B0, B1, B2, B3, B4, B5, B6,
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
                    xydiameter, maxNeuronNPerVolume);
                
                /*segment_extract_curvature(B0,B1,B2,B3,B4,B5,B6,sizex2,sizey2,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
	                NeuronCurvature, extractCurvatureBoxSize);*/
                NeuronCurvature += NeuronNAll[volumeFirstFrame[mu]+nu]*extractCurvatureBoxSize;
                
                NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            }
            
            // with B3 at nu=framesInVolume-3 (B6=Zero)
            int nu = framesInVolume - 3;
            
            A0 = ArrAA + (framesInVolume-6)*sizexy2;
            A1 = ArrAA + (framesInVolume-5)*sizexy2;
            A2 = ArrAA + (framesInVolume-4)*sizexy2;
            A3 = ArrAA + (framesInVolume-3)*sizexy2;
            A4 = ArrAA + (framesInVolume-2)*sizexy2;
            A5 = ArrAA + (framesInVolume-1)*sizexy2;
            A6 = Zero_uint16;
            
            B0 = ArrBB + (framesInVolume-6)*sizexy2;
            B1 = ArrBB + (framesInVolume-5)*sizexy2;
            B2 = ArrBB + (framesInVolume-4)*sizexy2;
            B3 = ArrBB + (framesInVolume-3)*sizexy2;
            B4 = ArrBB + (framesInVolume-2)*sizexy2;
            B5 = ArrBB + (framesInVolume-1)*sizexy2;
            B6 = Zero;
            NeuronXYin += NeuronNin;
            NeuronNin = NeuronNCandidatesVolume[nu];
            NeuronXYout += NeuronNAll[volumeFirstFrame[mu]+nu-1];
            
            segment_check2dcandidates_7planes(A0, A1, A2, A3, A4, A5, A6, //TODO AA
                    B0, B1, B2, B3, B4, B5, B6,
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
                    xydiameter, maxNeuronNPerVolume);
                    
            /*segment_extract_curvature(B0,B1,B2,B3,B4,B5,B6,sizex2,sizey2,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
	                NeuronCurvature, extractCurvatureBoxSize);*/
            NeuronCurvature += NeuronNAll[volumeFirstFrame[mu]+nu]*extractCurvatureBoxSize;
                    
            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            
            // with B3 at nu=framesInVolume-2 (B6=B5=Zero)
            nu = framesInVolume - 2;
            
            A0 = ArrAA + (framesInVolume-5)*sizexy2; //TODO AA
            A1 = ArrAA + (framesInVolume-4)*sizexy2;
            A2 = ArrAA + (framesInVolume-3)*sizexy2;
            A3 = ArrAA + (framesInVolume-2)*sizexy2;
            A4 = ArrAA + (framesInVolume-1)*sizexy2;
            A5 = Zero_uint16;
            A6 = Zero_uint16;
            
            B0 = ArrBB + (framesInVolume-5)*sizexy2;
            B1 = ArrBB + (framesInVolume-4)*sizexy2;
            B2 = ArrBB + (framesInVolume-3)*sizexy2;
            B3 = ArrBB + (framesInVolume-2)*sizexy2;
            B4 = ArrBB + (framesInVolume-1)*sizexy2;
            B5 = Zero;
            B6 = Zero;
            NeuronXYin += NeuronNin;
            NeuronNin = NeuronNCandidatesVolume[nu];
            NeuronXYout += NeuronNAll[volumeFirstFrame[mu]+nu-1];
            
            segment_check2dcandidates_7planes(A0, A1, A2, A3, A4, A5, A6, //TODO AA
                    B0, B1, B2, B3, B4, B5, B6,
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
                    xydiameter, maxNeuronNPerVolume);
                    
            /*segment_extract_curvature(B0,B1,B2,B3,B4,B5,B6,sizex2,sizey2,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
	                NeuronCurvature, extractCurvatureBoxSize);*/
            NeuronCurvature += NeuronNAll[volumeFirstFrame[mu]+nu]*extractCurvatureBoxSize;
                    
            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            
            // with B3 at nu=framesInVolume-1 (B6=Zero)
            nu = framesInVolume - 1;
            
            A0 = ArrAA + (framesInVolume-4)*sizexy2; //TODO AA
            A1 = ArrAA + (framesInVolume-3)*sizexy2;
            A2 = ArrAA + (framesInVolume-2)*sizexy2;
            A3 = ArrAA + (framesInVolume-1)*sizexy2;
            A4 = Zero_uint16;
            A5 = Zero_uint16;
            A6 = Zero_uint16;
            
            B0 = ArrBB + (framesInVolume-4)*sizexy2;
            B1 = ArrBB + (framesInVolume-3)*sizexy2;
            B2 = ArrBB + (framesInVolume-2)*sizexy2;
            B3 = ArrBB + (framesInVolume-1)*sizexy2;
            B4 = Zero;
            B5 = Zero;
            B6 = Zero;
            NeuronXYin += NeuronNin;
            NeuronNin = NeuronNCandidatesVolume[nu];
            NeuronXYout += NeuronNAll[volumeFirstFrame[mu]+nu-1];
            
            segment_check2dcandidates_7planes(A0, A1, A2, A3, A4, A5, A6, //TODO AA
                    B0, B1, B2, B3, B4, B5, B6,
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
                    xydiameter, maxNeuronNPerVolume);
                    
            /*segment_extract_curvature(B0,B1,B2,B3,B4,B5,B6,sizex2,sizey2,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu],
	                NeuronCurvature, extractCurvatureBoxSize);*/
            NeuronCurvature += NeuronNAll[volumeFirstFrame[mu]+nu]*extractCurvatureBoxSize;
                    
            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
        }
    }
    }
    
    delete[] Zero;
    delete[] Zero_uint16; //TODO AA
}

/*
TODO TRANSPARENT API AND GPU?
*/

void segment_singleframe_pipeline(uint16_t ImgIn[], 
	int32_t sizex, int32_t sizey, 
	cv::Mat& C, int32_t sizeC, 
	cv::Mat& A, cv::Mat& B, cv::Mat& BX, cv::Mat& BY, 
	cv::Mat& Bth, cv::Mat& Bdil, cv::Mat& K,
	uint32_t NeuronXY[], uint32_t &NeuronN, 
	double &maxX, double &maxY, double maxXInStack, double maxYInStack,
	int &chunk_tot_count, int chunk_tot_thresh,
	float threshold, double blur, uint16_t A_thresh, bool resize) {
	
	uint32_t k = 0;

	if (resize) {
		// Assume that both sizes of the image are even numbers.
		int sizex2 = sizex / 2;
		int sizey2 = sizey / 2;
		
		// Resize image
		cv::resize(cv::Mat(sizex, sizey, CV_16U, ImgIn), A, 
			A.size(), 0, 0, cv::INTER_LINEAR);//INTER_AREA
			
		cv::GaussianBlur(A, A, cv::Size(3, 3), blur, blur);
			
		// Determine if it's an empty image by counting the number of 
		// contiguous chunks above a threshold.
		int chunk = 4;
		int chunk_int_count;
		chunk_tot_count=0;
		for(int i=chunk;i<sizex2*sizey2;i++){
            chunk_int_count = 0;
		    for(int j=0;j<chunk;j++){
		        if(A.at<uint16_t>(i-j)>A_thresh+20){chunk_int_count+=1;}
	        }
	        if(chunk_int_count==chunk){chunk_tot_count+=1;}
		}
		
		
        if(chunk_tot_count>chunk_tot_thresh/10.0){
        
		// Dilate
		// 60 us
		for(int i=0;i<sizex2*sizey2;i++){
		    B.at<float>(i)=A.at<uint16_t>(i);    
		}
		cv::dilate(B, Bdil, K);
		
		// Do a 1-dimensional scan over the arrays. Faster than two nested for
		// loops. You will have to calculate the xy coordinates from the single
		// index i.
		// 60 us
		float tmpBi;
		float tmpBdili;
		float tmpAi;
		int n_A_above_thresh;
		
		for (int i = 0; i < sizex2*sizey2; i++) {
			tmpBi = B.at<float>(i);
			tmpBdili = Bdil.at<float>(i);
			tmpAi = A.at<uint16_t>(i);
			if (tmpBi == tmpBdili && tmpAi > A_thresh && k<302) {
			    n_A_above_thresh = 0;
			    if(i>sizex2 && i<sizex2*(sizey2-1)){
			        for(int m=0;m<3;++m){
			        for(int l=0;l<3;++l){
			            if(A.at<uint16_t>(i+(m-1)*sizex2+(l-1))>A_thresh){
			                n_A_above_thresh++;
			            }
			        }}
			        
			        /*if(A.at<uint16_t>(i-sizex2)>A_thresh){n_A_above_thresh++;}
			        if(A.at<uint16_t>(i-1)>A_thresh){n_A_above_thresh++;}
			        if(A.at<uint16_t>(i+1)>A_thresh){n_A_above_thresh++;}
			        if(A.at<uint16_t>(i+sizex2)>A_thresh){n_A_above_thresh++;}*/
		        }
		        if(n_A_above_thresh > 3){
			        NeuronXY[k] = i;
			        k++;
			    }
			}
		}
	
	    } else {
	    // just copy A in B
	        for(int i=0;i<sizex2*sizey2;i++){
		        B.at<float>(i)=A.at<uint16_t>(i);    
		    }
	    }
	}
	NeuronN = k;
}

void segment_singleframe_pipeline_old(uint16_t ImgIn[], 
	int32_t sizex, int32_t sizey, 
	cv::Mat& C, int32_t sizeC, 
	cv::Mat& A, cv::Mat& B, cv::Mat& BX, cv::Mat& BY, 
	cv::Mat& Bth, cv::Mat& Bdil, cv::Mat& K,
	uint32_t NeuronXY[], uint32_t &NeuronN, 
	double &maxX, double &maxY, double maxXInStack, double maxYInStack,
	int &chunk_tot_count, int chunk_tot_thresh,
	float threshold, double blur, uint16_t A_thresh, bool resize) {
	
	// The execution times below are for the lab computer:
	// Dual Intel Xeon Processor E5-2643 v4 
	// (6C, 3.4GHz, 3.7GHz Turbo, 2400MHz, 20MB, 135W);
	// 64GB (8x8GB) 2400MHz DDR4 RDIMM ECC;

	uint32_t k = 0;

	if (resize) {
		// Assume that both sizes of the image are even numbers.
		int sizex2 = sizex / 2;
		int sizey2 = sizey / 2;
		
		cv::Mat OneHalf = cv::Mat(1, 1, CV_32F, cv::Scalar::all(0.5));
		//cv::Mat K = cv::Mat(5, 5, CV_32F, cv::Scalar::all(1));
		
		// Other variables.
		double minX, minY, threshX, threshY;

		// Resize image
		cv::resize(cv::Mat(sizex, sizey, CV_16U, ImgIn), A, 
			A.size(), 0, 0, cv::INTER_AREA);
			
		// Determine if it's an empty image by counting the number of 
		// contiguous chunks above a threshold.
		//uint16_t A_thresh = 110;
		int chunk = 4;
		int chunk_int_count;
		chunk_tot_count=0;
		for(int i=chunk;i<sizex2*sizey2;i++){
            chunk_int_count = 0;
		    for(int j=0;j<chunk;j++){
		        if(A.at<uint16_t>(i-j)>A_thresh+20){chunk_int_count+=1;}
	        }
	        if(chunk_int_count==chunk){chunk_tot_count+=1;}
		}
        
        if(chunk_tot_count>chunk_tot_thresh/10.0){
        
		// Apply Gaussian blur
		// 210 us
		//double blur = 0.65; //0.65
		cv::GaussianBlur(A, A, cv::Size(5, 5), blur, blur);
		
		
		//Gamma correction			
        /*cv::Mat Ag = cv::Mat(sizex2, sizey2, CV_32F, cv::Scalar::all(0.0));
        for(int pq=0;pq<sizex2*sizey2;pq++){
            Ag.at<float>(pq) = pow(std::min(std::max(A.at<uint16_t>(pq)-110.,0.0),2000.),0.5);
        }*/
        
		// Calculate -d2/dx2 and -d2/dy2 with the filter passed as C.
		// In this way, the CV_32F images will saturate the positive 
		// derivatives and not the negative ones, which we need.
		// Before summing the two derivatives, they need to be divided by 2, 
		// so that they don't saturate. Instead of separately divide them,
		// use 1/2 instead of the 1 as second filter in the separable filter
		// function. (Saves 100 microseconds)
		// 300 us
        
		cv::sepFilter2D(A, BX, CV_32F, C, OneHalf); //FIXME Ag. If you want the gamma correction 
		cv::sepFilter2D(A, BY, CV_32F, OneHalf, C); //to be active replace A with Ag.
		cv::add(BX,BY,B);
		
		// Threshold out some noise below a fraction of the maximum values:
		// Keep only the points in which **both** derivatives are above
		// threshold separately
		// (remember that BX and BY are -d2/dx2 and -d2/dy2).
		// The maxima used to calculate the threshold are either the one from
		// the current frame, or an estimate of the maximum in the stack
		// calculated from the previous stack (something like the n-th brightest
		// Hessian, so that you don't get too large threshold because of very
		// shar
		// 100 us
		/**cv::minMaxIdx(BX, &minX, &maxX, NULL, NULL);
		cv::minMaxIdx(BY, &minY, &maxY, NULL, NULL);
		
		////This was before 30th April 2021 change
		//maxX = kthlargest((float*)BX.data, sizex2*sizey2, 10); //10
        //maxY = kthlargest((float*)BY.data, sizex2*sizey2, 10); //10
		if ( (maxXInStack == -1.0 && maxYInStack == -1.0) || 
		     (maxXInStack < maxX || maxYInStack < maxY) ) {
			threshX = threshold * maxX;
			threshY = threshold * maxY;
		} else {
			threshX = threshold * maxXInStack;
			threshY = threshold * maxYInStack;
		}
		//cv::threshold(BX, BX, threshX, 1, CV_THRESH_BINARY);
		//cv::threshold(BY, BY, threshY, 1, CV_THRESH_BINARY);
		//Bth = BX.mul(BY);
		cv::threshold(B, Bth, 0.5*(threshX+threshY), 1, CV_THRESH_BINARY);
		Bth = Bth.mul(B);**/
		for (int i=0;i<sizex2*sizey2;++i){
		    Bth.at<float>(i) = B.at<float>(i);
		}
		
		/**
		//FIXME JUST DOING WITH AA As
		for(int pq=0;pq<sizex2*sizey2;pq++){
		    BX.at<float>(pq) = Ag.at<float>(pq)/2;
		    BY.at<float>(pq) = Ag.at<float>(pq)/2;
		    B.at<float>(pq) = Ag.at<float>(pq);
		    Bth.at<float>(pq) = Ag.at<float>(pq);
		}**/
        
		// Dilate
		// 60 us
		/**for(int i=0;i<sizex2*sizey2;i++){
		    Bth.at<float>(i)=A.at<uint16_t>(i);
		    B.at<float>(i)=A.at<uint16_t>(i);    
		}**/
		cv::dilate(Bth, Bdil, K);
		
		// Do a 1-dimensional scan over the arrays. Faster than two nested for
		// loops. You will have to calculate the xy coordinates from the single
		// index i.
		// 60 us
		float tmpBi;
		float tmpBdili;
		float tmpAi;
		uint16_t A_neigh[25];
		uint16_t i_max_subset, i_max;
		int n_A_above_thresh;
		
		for (int i = 0; i < sizex2*sizey2; i++) {
			tmpBi = B.at<float>(i);
			tmpBdili = Bdil.at<float>(i);
			tmpAi = A.at<uint16_t>(i);
			n_A_above_thresh = 0;
			if (tmpBi != 0.0 && tmpBi == tmpBdili && tmpAi > A_thresh && k<302) {
			    /**if(i>sizex2 && i<sizex2*(sizey2-1)){
			        if(A.at<uint16_t>(i-sizex2)>A_thresh){n_A_above_thresh+=1;}
			        if(A.at<uint16_t>(i-1)>A_thresh){n_A_above_thresh+=1;}
			        if(A.at<uint16_t>(i)>A_thresh){n_A_above_thresh+=1;}
			        if(A.at<uint16_t>(i+1)>A_thresh){n_A_above_thresh+=1;}
			        if(A.at<uint16_t>(i+sizex2)>A_thresh){n_A_above_thresh+=1;}
		        }
		        if(n_A_above_thresh > 4){
			        NeuronXY[k] = i;
			        k++;
			    }**/
			    NeuronXY[k] = i;
		        k++;
			    /**FIND ACTUAL MAX IN A
			    if(i<(sizex2*(sizey2-2)) && i%sizex2<(sizex2-2) && i%sizex2>2){
			        for(int q=-2;q<=2;q++) {
			            for(int p=-2;p<=2;p++){
			                A_neigh[(q+2)*5+(p+2)] = A.at<uint16_t>(i+q*sizex2+p);
			            }
			        }
			        // Find index of maximum value in the neighborhood, first
			        // as index in the neighborhood subset and then as index
			        // in the full image array.
			        i_max_subset = std::distance(A_neigh, std::max_element(A_neigh , A_neigh + 25));
			        i_max = i+sizex2*(i_max_subset/5)+i_max_subset%5;
			        // Store the index as a neuron candidate in this plane.
			        NeuronXY[k] = i_max;
			        END FIND ACTUAL MAX IN A**/
			        
			        // THIS TO EXTRACT THE POSITION BASED ON CURVATURE ALONE
			    
				//}
			}
		}
	
	    } else {
	    // just copy A in B
	        for(int i=0;i<sizex2*sizey2;i++){
		        B.at<float>(i)=0.0;//A.at<uint16_t>(i);    
		    }
	    }
	}
	NeuronN = k;
}

void segment_check2dcandidates_5planes(
	float ArrB0[], float ArrB1[], float ArrB2[],
	float ArrB3[], float ArrB4[], int32_t sizeBx, int32_t sizeBy, 
	uint32_t NeuronXYin[], uint32_t NeuronNin,
	uint32_t *NeuronXYout, uint32_t &NeuronNout,
	int32_t maxNeuronNPerVolume) {
	//uint32_t *NeuronNAll, uint32_t *volumeFirstFrame, bool check_if_other_candidate

	// Brute force check if the candidate neurons found on each plane are
	// actually maxima in -d2/dx2-d2/dy2 also in a sphere around them and not
	// just in their plane.
	// This function can run as soon as the 5 planes are available, so that
	// you don't have to wait until the full volume is ready before starting
	// this brute force check.
	
	float Bxy;
	int32_t k = 0;
	uint32_t index;
	uint32_t upperlimit = sizeBx*sizeBy - sizeBx;

	for (uint i = 0; i < NeuronNin; i++) {
		index = NeuronXYin[i];
		Bxy = ArrB2[index];

		// I will hard code the regions where to look.
		// Since B is -d2/dx2-d2/dy2, I have to check if it's the maximum.
		
		///////////////////////////////
		///// - maybe do a for loop over the y-1,y,y+1, so that the compiler
		/////	can vectorize at least that.
		///////////////////////////////
/**
		//ArrB0 and ArrB4 (single pixel)
		if (Bxy < ArrB0[index]) ok = false;
		if (Bxy < ArrB4[index]) ok = false;

		//ArrB1 and ArrB3 (cross shaped ROI)
		if (Bxy < ArrB1[index - sizeBx]) ok = false;
		if (Bxy < ArrB1[index - 1]) ok = false;
		if (Bxy < ArrB1[index]) ok = false;
		if (Bxy < ArrB1[index + 1]) ok = false;
		if (Bxy < ArrB1[index + sizeBx]) ok = false;

		if (Bxy < ArrB3[index - sizeBx]) ok = false;
		if (Bxy < ArrB3[index - 1]) ok = false;
		if (Bxy < ArrB3[index]) ok = false;
		if (Bxy < ArrB3[index + 1]) ok = false;
		if (Bxy < ArrB3[index + sizeBx]) ok = false;
**/		if (index > (uint32_t)sizeBx && index < upperlimit && k < maxNeuronNPerVolume) {

            /**if(check_if_other_candidate){
                ok2 = true;
                for(int l=-2;l<2;l++){
                for(int q=0;q<NeuronNAll[*(volumeFirstFrame+l)];q++){
                    if(NeuronXYAll[
                }}
            }**/
            
		    if (    (Bxy > ArrB0[index]) &&
		            (Bxy > ArrB4[index]) &&
		            (Bxy > ArrB1[index - sizeBx]) &&
		            (Bxy > ArrB1[index - 1]) &&
		            (Bxy > ArrB1[index]) &&
		            (Bxy > ArrB1[index + 1]) &&
		            (Bxy > ArrB1[index + sizeBx]) &&
		            (Bxy > ArrB3[index - sizeBx]) &&
		            (Bxy > ArrB3[index - 1]) &&
		            (Bxy > ArrB3[index]) &&
		            (Bxy > ArrB3[index + 1]) &&
		            (Bxy > ArrB3[index + sizeBx]) ) {
			    *(NeuronXYout+k) = index; 
			    k++;
		    }
		}
	}
	NeuronNout = k;
}

void segment_check2dcandidates_7planes(
    uint16_t ArrA0[], uint16_t ArrA1[], uint16_t ArrA2[],
    uint16_t ArrA3[], uint16_t ArrA4[], uint16_t ArrA5[], uint16_t ArrA6[],
	float ArrB0[], float ArrB1[], float ArrB2[],
	float ArrB3[], float ArrB4[], float ArrB5[], float ArrB6[],
    int32_t sizeBx, int32_t sizeBy, 
	uint32_t NeuronXYin[], uint32_t NeuronNin,
	uint32_t *NeuronXYout, uint32_t &NeuronNout, uint32_t maxdiameter,
	int32_t maxNeuronNPerVolume) {
    
    // Alias for one of the two versions
    
    if(maxdiameter==3){
        segment_check2dcandidates_7planes_3maxdiameter(
        ArrA0,ArrA1,ArrA2,ArrA3,ArrA4,ArrA5,ArrA6, //TODO AA
        ArrB0,ArrB1,ArrB2,ArrB3,ArrB4,ArrB5,ArrB6,
        sizeBx,sizeBy,NeuronXYin,NeuronNin,NeuronXYout,NeuronNout,maxNeuronNPerVolume);
    } else if(maxdiameter==5){
        segment_check2dcandidates_7planes_5maxdiameter(
        ArrA0,ArrA1,ArrA2,ArrA3,ArrA4,ArrA5,ArrA6, //TODO AA
        ArrB0,ArrB1,ArrB2,ArrB3,ArrB4,ArrB5,ArrB6,
        sizeBx,sizeBy,NeuronXYin,NeuronNin,NeuronXYout,NeuronNout,maxNeuronNPerVolume);
    }
	
}


void segment_check2dcandidates_7planes_3maxdiameter(
    uint16_t ArrA0[], uint16_t ArrA1[], uint16_t ArrA2[],
    uint16_t ArrA3[], uint16_t ArrA4[], uint16_t ArrA5[], uint16_t ArrA6[],
	float ArrB0[], float ArrB1[], float ArrB2[],
	float ArrB3[], float ArrB4[], float ArrB5[], float ArrB6[],
    int32_t sizeBx, int32_t sizeBy, 
	uint32_t NeuronXYin[], uint32_t NeuronNin,
	uint32_t *NeuronXYout, uint32_t &NeuronNout,
	int32_t maxNeuronNPerVolume) {

	// Brute force check if the candidate neurons found on each plane are
	// actually maxima in -d2/dx2-d2/dy2 also in a sphere around them and not
	// just in their plane.
	// This function can run as soon as the 7 planes are available, so that
	// you don't have to wait until the full volume is ready before starting
	// this brute force check.
	
	uint16_t Axy; //TODO AA
	float Bxy;
	int32_t k = 0;
	uint32_t index;
	uint32_t upperlimit = sizeBx*sizeBy - sizeBx;

	for (uint i = 0; i < NeuronNin; i++) {
		index = NeuronXYin[i];
		Axy = ArrA3[index]; //TODO AA
		Bxy = ArrB3[index];

		if (index > (uint32_t)sizeBx && index < upperlimit && k < maxNeuronNPerVolume) {
		    if (    (Bxy > ArrB0[index]) &&
		    
		            (Bxy > ArrB6[index]) &&
		            
		            (Bxy > ArrB1[index - sizeBx]) &&
		            (Bxy > ArrB1[index - 1]) &&
		            (Bxy > ArrB1[index]) &&
		            (Bxy > ArrB1[index + 1]) &&
		            (Bxy > ArrB1[index + sizeBx]) &&
		            
		            (Bxy > ArrB2[index - 1]) &&
		            (Bxy > ArrB2[index]) &&
		            (Bxy > ArrB2[index + 1]) &&
		            (Bxy > ArrB2[index + sizeBx]) &&
		            
		            (Bxy > ArrB4[index - sizeBx]) &&
		            (Bxy > ArrB4[index - 1]) &&
		            (Bxy > ArrB4[index]) &&
		            (Bxy > ArrB4[index + 1]) &&
		            (Bxy > ArrB4[index + sizeBx]) &&
		            
		            (Bxy > ArrB5[index - 1]) &&
		            (Bxy > ArrB5[index]) &&
		            (Bxy > ArrB5[index + 1]) &&
		            (Bxy > ArrB5[index + sizeBx]) /**&&
		            
		            //As

		            (Axy > ArrA0[index]) &&
		    
		            (Axy > ArrA6[index]) &&
		            
		            (Axy > ArrA1[index - sizeBx]) &&
		            (Axy > ArrA1[index - 1]) &&
		            (Axy > ArrA1[index]) &&
		            (Axy > ArrA1[index + 1]) &&
		            (Axy > ArrA1[index + sizeBx]) &&
		            
		            (Axy > ArrA2[index - 1]) &&
		            (Axy > ArrA2[index]) &&
		            (Axy > ArrA2[index + 1]) &&
		            (Axy > ArrA2[index + sizeBx]) &&
		            
		            (Axy > ArrA4[index - sizeBx]) &&
		            (Axy > ArrA4[index - 1]) &&
		            (Axy > ArrA4[index]) &&
		            (Axy > ArrA4[index + 1]) &&
		            (Axy > ArrA4[index + sizeBx]) &&
		            
		            (Axy > ArrA5[index - 1]) &&
		            (Axy > ArrA5[index]) &&
		            (Axy > ArrA5[index + 1]) &&
		            (Axy > ArrA5[index + sizeBx])**/ ) {
			    *(NeuronXYout+k) = index; 
			    k++;
		    }
		}
	}
	NeuronNout = k;
}

void segment_check2dcandidates_7planes_5maxdiameter(
    uint16_t ArrA0[], uint16_t ArrA1[], uint16_t ArrA2[],
    uint16_t ArrA3[], uint16_t ArrA4[], uint16_t ArrA5[], uint16_t ArrA6[],
	float ArrB0[], float ArrB1[], float ArrB2[],
	float ArrB3[], float ArrB4[], float ArrB5[], float ArrB6[],
    int32_t sizeBx, int32_t sizeBy, 
	uint32_t NeuronXYin[], uint32_t NeuronNin,
	uint32_t *NeuronXYout, uint32_t &NeuronNout,
	int32_t maxNeuronNPerVolume) {

	// Brute force check if the candidate neurons found on each plane are
	// actually maxima in -d2/dx2-d2/dy2 also in a sphere around them and not
	// just in their plane.
	// This function can run as soon as the 7 planes are available, so that
	// you don't have to wait until the full volume is ready before starting
	// this brute force check.
	
	float Bxy;
	int32_t k = 0;
	uint32_t index;
	uint32_t upperlimit = sizeBx*sizeBy - 2*sizeBx;

	for (uint i = 0; i < NeuronNin; i++) {
		index = NeuronXYin[i];
		Bxy = ArrB3[index];

		if (index > 2*((uint32_t)sizeBx) && index < upperlimit && k < maxNeuronNPerVolume) {
		    if (    (Bxy > ArrB0[index]) &&
		    
		            (Bxy > ArrB6[index]) &&
		            
		            (Bxy > ArrB1[index - sizeBx]) &&
		            (Bxy > ArrB1[index - 1]) &&
		            (Bxy > ArrB1[index]) &&
		            (Bxy > ArrB1[index + 1]) &&
		            (Bxy > ArrB1[index + sizeBx]) &&
		            
		            (Bxy > ArrB2[index - 2*sizeBx]) &&
		            (Bxy > ArrB2[index - sizeBx-1]) &&
		            (Bxy > ArrB2[index - sizeBx]) &&
		            (Bxy > ArrB2[index - sizeBx+1]) &&
		            (Bxy > ArrB2[index - 2]) &&
		            (Bxy > ArrB2[index - 1]) &&
		            (Bxy > ArrB2[index]) &&
		            (Bxy > ArrB2[index + 1]) &&
		            (Bxy > ArrB2[index + 2]) &&
		            (Bxy > ArrB2[index + sizeBx-1]) &&
		            (Bxy > ArrB2[index + sizeBx]) &&
		            (Bxy > ArrB2[index + sizeBx+1]) &&
		            (Bxy > ArrB2[index + 2*sizeBx]) &&
		            
		            (Bxy > ArrB4[index - 2*sizeBx]) &&
		            (Bxy > ArrB4[index - sizeBx-1]) &&
		            (Bxy > ArrB4[index - sizeBx]) &&
		            (Bxy > ArrB4[index - sizeBx+1]) &&
		            (Bxy > ArrB4[index - 2]) &&
		            (Bxy > ArrB4[index - 1]) &&
		            (Bxy > ArrB4[index]) &&
		            (Bxy > ArrB4[index + 1]) &&
		            (Bxy > ArrB4[index + 2]) &&
		            (Bxy > ArrB4[index + sizeBx-1]) &&
		            (Bxy > ArrB4[index + sizeBx]) &&
		            (Bxy > ArrB4[index + sizeBx+1]) &&
		            (Bxy > ArrB4[index + 2*sizeBx]) &&

		            
		            (Bxy > ArrB5[index - 1]) &&
		            (Bxy > ArrB5[index]) &&
		            (Bxy > ArrB5[index + 1]) &&
		            (Bxy > ArrB5[index + sizeBx]) ) {
			    *(NeuronXYout+k) = index; 
			    k++;
		    }
		}
	}
	NeuronNout = k;
}

void segment_extract_curvature(
	float ArrB0[], float ArrB1[], float ArrB2[],
	float ArrB3[], float ArrB4[], float ArrB5[], float ArrB6[],
    int32_t sizeBx, int32_t sizeBy, 
	uint32_t NeuronXYin[], uint32_t NeuronNin,
	float *NeuronCurvatureOut, uint32_t totalBoxSize) {

    /**
    Parameters
    ----------
    ArrB0, ArrB1, ArrB2, ArrB3, ArrB4, ArrB5, ArrB6 : float arrays
        Arrays containing the curvature in the current plane and the
        neighboring ones.
    sizeBx, sizeBy : int32_t
        Sizes of the single images.
    NeuronXYin : uint32_t array
        Array containing the linearized indexes of the (confirmed) neurons. The
        indexes are relative to the frame in which the neuron has been found.
    NeuronNin : uint32_t
        Number of neurons in the current frame (the one corresponding to the
        curvature B3).
    NeuronCurvatureOut : pointer to float
        Pointing to the array storing the curvatures, at the first free
        position. After calling this function, this pointer has to be moved by
        NeuronNIn*totalBoxSize, so that the next time it is passed to this
        function, it will point to the first free positions.
    
    Returns
    -------
    void.
    
    Notes
    -----
    Given the already *confirmed* neurons (not all the candidates), save the
    curvatures of the neighboring points. For the time being, hard code a box.
    This works like a kind of watershed, giving you as a result the extension 
    of the neuron. 
    
    Currently using the "7x5" box of the check 7 planes 5 max diameter.
	**/
	
	int32_t k = 0;
	uint32_t index;
    
    if(totalBoxSize==51){
	for (uint i = 0; i < NeuronNin; i++) {
	    index = NeuronXYin[i];
	    if(index>2*sizeBx && index<(sizeBx-2)*sizeBy) {
		*(NeuronCurvatureOut+k+0) = ArrB0[index];
		
		*(NeuronCurvatureOut+k+1)  = ArrB1[index - sizeBx];
		*(NeuronCurvatureOut+k+2)  = ArrB1[index - 1];
		*(NeuronCurvatureOut+k+3)  = ArrB1[index];
		*(NeuronCurvatureOut+k+4)  = ArrB1[index + 1];
		*(NeuronCurvatureOut+k+5)  = ArrB1[index + sizeBx];
		
		*(NeuronCurvatureOut+k+6)  = ArrB2[index - 2*sizeBx];
		*(NeuronCurvatureOut+k+7)  = ArrB2[index - sizeBx-1];
		*(NeuronCurvatureOut+k+8)  = ArrB2[index - sizeBx];
		*(NeuronCurvatureOut+k+9)  = ArrB2[index - sizeBx+1];
		*(NeuronCurvatureOut+k+10) = ArrB2[index - 2];
		*(NeuronCurvatureOut+k+11) = ArrB2[index - 1];
		*(NeuronCurvatureOut+k+12) = ArrB2[index];
		*(NeuronCurvatureOut+k+13) = ArrB2[index + 1];
		*(NeuronCurvatureOut+k+14) = ArrB2[index + 2];
		*(NeuronCurvatureOut+k+15) = ArrB2[index + sizeBx-1];
		*(NeuronCurvatureOut+k+16) = ArrB2[index + sizeBx];
		*(NeuronCurvatureOut+k+17) = ArrB2[index + sizeBx+1];
		*(NeuronCurvatureOut+k+18) = ArrB2[index + 2*sizeBx];
		
		*(NeuronCurvatureOut+k+19) = ArrB3[index - 2*sizeBx];
		*(NeuronCurvatureOut+k+20) = ArrB3[index - sizeBx-1];
		*(NeuronCurvatureOut+k+21) = ArrB3[index - sizeBx];
		*(NeuronCurvatureOut+k+22) = ArrB3[index - sizeBx+1];
		*(NeuronCurvatureOut+k+23) = ArrB3[index - 2];
		*(NeuronCurvatureOut+k+24) = ArrB3[index - 1];
		*(NeuronCurvatureOut+k+25) = ArrB3[index];
		*(NeuronCurvatureOut+k+26) = ArrB3[index + 1];
		*(NeuronCurvatureOut+k+27) = ArrB3[index + 2];
		*(NeuronCurvatureOut+k+28) = ArrB3[index + sizeBx-1];
		*(NeuronCurvatureOut+k+29) = ArrB3[index + sizeBx];
		*(NeuronCurvatureOut+k+30) = ArrB3[index + sizeBx+1];
		*(NeuronCurvatureOut+k+31) = ArrB3[index + 2*sizeBx];
		
		*(NeuronCurvatureOut+k+32) = ArrB4[index - 2*sizeBx];
		*(NeuronCurvatureOut+k+33) = ArrB4[index - sizeBx-1];
		*(NeuronCurvatureOut+k+34) = ArrB4[index - sizeBx];
		*(NeuronCurvatureOut+k+35) = ArrB4[index - sizeBx+1];
		*(NeuronCurvatureOut+k+36) = ArrB4[index - 2];
		*(NeuronCurvatureOut+k+37) = ArrB4[index - 1];
		*(NeuronCurvatureOut+k+38) = ArrB4[index];
		*(NeuronCurvatureOut+k+39) = ArrB4[index + 1];
		*(NeuronCurvatureOut+k+40) = ArrB4[index + 2];
		*(NeuronCurvatureOut+k+41) = ArrB4[index + sizeBx-1];
		*(NeuronCurvatureOut+k+42) = ArrB4[index + sizeBx];
		*(NeuronCurvatureOut+k+43) = ArrB4[index + sizeBx+1];
		*(NeuronCurvatureOut+k+44) = ArrB4[index + 2*sizeBx];
		
		*(NeuronCurvatureOut+k+45) = ArrB5[index - sizeBx];
		*(NeuronCurvatureOut+k+46) = ArrB5[index - 1];
		*(NeuronCurvatureOut+k+47) = ArrB5[index];
		*(NeuronCurvatureOut+k+48) = ArrB5[index + 1];
		*(NeuronCurvatureOut+k+49) = ArrB5[index + sizeBx];
		
		*(NeuronCurvatureOut+k+50) = ArrB6[index];
	    } else {
	        for(int curvi=0;curvi<51;curvi++){
	            *(NeuronCurvatureOut+k+curvi) = 1;
	        }
	    }
	    
		k += totalBoxSize;
	}
	}
}


void segment_extract_curvature_single_frame(
	float ArrB[],
    int32_t sizeBx, int32_t sizeBy, 
	uint32_t NeuronXYin[], uint32_t NeuronNin,
	float *NeuronCurvatureOut, uint32_t totalBoxSize) {

    /**
    Parameters
    ----------
    ArrB0, ArrB1, ArrB2, ArrB3, ArrB4, ArrB5, ArrB6 : float arrays
        Arrays containing the curvature in the current plane and the
        neighboring ones.
    sizeBx, sizeBy : int32_t
        Sizes of the single images.
    NeuronXYin : uint32_t array
        Array containing the linearized indexes of the (confirmed) neurons. The
        indexes are relative to the frame in which the neuron has been found.
    NeuronNin : uint32_t
        Number of neurons in the current frame (the one corresponding to the
        curvature B3).
    NeuronCurvatureOut : pointer to float
        Pointing to the array storing the curvatures, at the first free
        position. After calling this function, this pointer has to be moved by
        NeuronNIn*totalBoxSize, so that the next time it is passed to this
        function, it will point to the first free positions.
    
    Returns
    -------
    void.
    
    Notes
    -----
    Given the already *confirmed* neurons (not all the candidates), save the
    curvatures of the neighboring points. For the time being, hard code a box.
    This works like a kind of watershed, giving you as a result the extension 
    of the neuron. 
    
    Currently using the "7x5" box of the check 7 planes 5 max diameter.
	**/
	
	int32_t k = 0;
	uint32_t index;
    
    if(totalBoxSize==13){
	for (uint i = 0; i < NeuronNin; i++) {
	    index = NeuronXYin[i];
		
		*(NeuronCurvatureOut+k+0) = ArrB[index - 2*sizeBx];
		*(NeuronCurvatureOut+k+1) = ArrB[index - sizeBx-1];
		*(NeuronCurvatureOut+k+2) = ArrB[index - sizeBx];
		*(NeuronCurvatureOut+k+3) = ArrB[index - sizeBx+1];
		*(NeuronCurvatureOut+k+4) = ArrB[index - 2];
		*(NeuronCurvatureOut+k+5) = ArrB[index - 1];
		*(NeuronCurvatureOut+k+6) = ArrB[index];
		*(NeuronCurvatureOut+k+7) = ArrB[index + 1];
		*(NeuronCurvatureOut+k+8) = ArrB[index + 2];
		*(NeuronCurvatureOut+k+9) = ArrB[index + sizeBx-1];
		*(NeuronCurvatureOut+k+10) = ArrB[index + sizeBx];
		*(NeuronCurvatureOut+k+11) = ArrB[index + sizeBx+1];
		*(NeuronCurvatureOut+k+12) = ArrB[index + 2*sizeBx];
		
		k += totalBoxSize;
	}
	}
}

float kthlargest(float array[], int size, int k) {
    std::priority_queue<std::pair<float, int>> q;
    for (int i = 0; i < size; ++i) {
        q.push(std::pair<float, int>(array[i], i));
    }
    int ki;
    for (int i = 0; i < k; ++i) {
        ki = q.top().second;
        q.pop();
    }
    return array[ki];
}
