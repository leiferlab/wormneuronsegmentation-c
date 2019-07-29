#include <stdint.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "neuronsegmentation.h"
#include <iostream>

void find_neurons_frames_sequence(uint16_t framesIn[],
    uint32_t framesN, int32_t sizex, int32_t sizey,
    int32_t framesStride, // 1 or 2 (RFP RFP RFP or RFP GFP RFP GFP)
    uint16_t ArrA[], 
    float ArrB[], float ArrBX[], float ArrBY[], 
	float ArrBth[], float ArrBdil[],
	uint32_t NeuronXY[], uint32_t NeuronN[],
	float threshold, double blur
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
       
    // Create the cv::Mat header for all the images.
    cv::Mat A = cv::Mat(sizex2, sizey2, CV_16U, ArrA);
    cv::Mat B = cv::Mat(sizex2, sizey2, CV_32F, ArrB);
	cv::Mat BX = cv::Mat(sizex2, sizey2, CV_32F, ArrBX);
	cv::Mat BY = cv::Mat(sizex2, sizey2, CV_32F, ArrBY);
	cv::Mat Bth = cv::Mat(sizex2, sizey2, CV_32F, ArrBth);
	cv::Mat Bdil = cv::Mat(sizex2, sizey2, CV_32F, ArrBdil);
	cv::Mat C = cv::Mat(sizeC, 1, CV_32F, ArrC);
	cv::Mat OneHalf = cv::Mat(1, 1, CV_32F, cv::Scalar::all(0.5));
	cv::Mat K = cv::Mat(3, 3, CV_32F, cv::Scalar::all(1));
	    
    double maxX, maxY, maxXInStack, maxYInStack;
    //float threshold = 0.25;//0.25
            
    // Index for the frames with respect to the beginning
    ImgIn = framesIn;
    
    maxXInStack = -1.0;
    maxYInStack = -1.0;
        
    // Run the single frame segmentation to initialize maxX/YInStack
    segment_singleframe_pipeline(ImgIn, sizex, sizey, 
        C, sizeC, A, B, BX, BY, Bth, Bdil, 
        NeuronXY, NeuronN[0],
        maxX, maxY, maxXInStack, maxYInStack, threshold, blur, true);
        
    maxXInStack = maxX;
    maxYInStack = maxY;
        
    // Segment each frame independently and find the neurons.
    for(uint nu=0; nu<framesN; nu++) {
            //maxXInStack = -1.0;
            //maxYInStack = -1.0;
              
        // Run the single frame segmentation.
        segment_singleframe_pipeline(ImgIn, sizex, sizey, 
            C, sizeC, A, B, BX, BY, Bth, Bdil, 
            NeuronXY, NeuronN[nu],
            maxX, maxY, maxXInStack, maxYInStack, threshold, blur, true);
        
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
            
            // Update encountered maxima.
            maxXInStack = 0.7*((maxXInStack<maxX)?maxX:maxXInStack);
            maxYInStack = 0.7*((maxYInStack<maxY)?maxY:maxYInStack);
        }
    }
}

void find_neurons(uint16_t framesIn[],
    uint32_t framesN, int32_t sizex, int32_t sizey,
    int32_t framesStride, // 1 or 2 (RFP RFP RFP or RFP GFP RFP GFP)
    uint32_t volumeFirstFrame[], uint32_t volumeN,
    uint16_t ArrA[], 
    float ArrBB[], float ArrBX[], float ArrBY[], 
	float ArrBth[], float ArrBdil[],
	uint32_t NeuronXYCandidatesVolume[], 
	uint32_t NeuronNCandidatesVolume[],
	uint32_t NeuronXYAll[], uint32_t NeuronNAll[],
	float threshold, double blur, uint32_t checkPlanesN
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
    
    // Zero is used later in the selection of the candidate neurons.   
    float * Zero;
    Zero = new float[sizexy2];
    for(int k=0; k<sizexy2; k++) { Zero[k] = 0.0; }
    
    // Pointer to ImgIn inside framesIn[]
    uint16_t * ImgIn;
    
    // Pointer to ArrB inside ArrBB
    float *ArrB;
    
    // Pointers to B0, B1, B2, B3, B4 for the candidate check.
    float *B0, *B1, *B2, *B3, *B4;
    
    // Create the cv::Mat header for all the images but B, which will need to 
    // be changed at every frame within a volume, so that they can be stored.
    cv::Mat A = cv::Mat(sizex2, sizey2, CV_16U, ArrA);
	cv::Mat BX = cv::Mat(sizex2, sizey2, CV_32F, ArrBX);
	cv::Mat BY = cv::Mat(sizex2, sizey2, CV_32F, ArrBY);
	cv::Mat Bth = cv::Mat(sizex2, sizey2, CV_32F, ArrBth);
	cv::Mat Bdil = cv::Mat(sizex2, sizey2, CV_32F, ArrBdil);
	cv::Mat C = cv::Mat(sizeC, 1, CV_32F, ArrC);
	cv::Mat OneHalf = cv::Mat(1, 1, CV_32F, cv::Scalar::all(0.5));
	cv::Mat K = cv::Mat(3, 3, CV_32F, cv::Scalar::all(1));
	
	cv::Mat B;
       
    double maxX, maxY, maxXInStack, maxYInStack;
    //float threshold = 0.25;
    
    maxXInStack = -1.0;
    maxYInStack = -1.0;
    
    // Index for the frames with respect to the beginning
    int i = 0;
    ImgIn = framesIn;
    
    // Point to the first allocated B frame in ArrBB. The pointer ArrB will
    // be moved as the analysis proceeds through the volume.
    ArrB = ArrBB;
    
    // Same as above
    NeuronXYCandidates = NeuronXYCandidatesVolume;
    
    
    // Initialize max_InStack with the first volume
    maxXInStack = -1.0;
    maxYInStack = -1.0;
    
    int framesInVolume = volumeFirstFrame[1] - volumeFirstFrame[0];
    
    // Segment each frame nu independently and find the candidate neurons.
    for(int nu=0; nu<framesInVolume; nu++) {
        
        // Create the Mat header for the pointer ArrB, that changes at every
        // iteration.
        B = cv::Mat(sizex2, sizey2, CV_32F, ArrB);
        
        // With the adjusted pointers and Mat headers, run the single frame
        // segmentation.
        segment_singleframe_pipeline(ImgIn, sizex, sizey, 
            C, sizeC, A, B, BX, BY, Bth, Bdil, 
            NeuronXYCandidates, NeuronNCandidatesVolume[nu],
            maxX, maxY, maxXInStack, maxYInStack, threshold, blur, true);
        
        // Move pointer to frames to the next one
        ImgIn = ImgIn + sizexy*framesStride;
                
        // Update encountered maxima.
        maxXInStack = (maxXInStack<maxX)?maxX:maxXInStack;
        maxYInStack = (maxYInStack<maxY)?maxY:maxYInStack;
    }
    
    // Bring pointer to frames to the beginning for the actual scan
    ImgIn = framesIn;
   
    // For each volume mu
    for(uint mu=0; mu<volumeN; mu++) {
        //maxXInStack = -1.0;
        //maxYInStack = -1.0;
        
        // Point to the first allocated B frame in ArrBB. The pointer ArrB will
        // be moved as the analysis proceeds through the volume.
        ArrB = ArrBB;
        
        // Same as above
        NeuronXYCandidates = NeuronXYCandidatesVolume;
        //NeuronXYCandidates = NeuronXYAll;
        
        int framesInVolume = volumeFirstFrame[mu+1] - volumeFirstFrame[mu];
        
        // Segment each frame nu independently and find the candidate neurons.
        for(int nu=0; nu<framesInVolume; nu++) {
            
            // Create the Mat header for the pointer ArrB, that changes at every
            // iteration.
            B = cv::Mat(sizex2, sizey2, CV_32F, ArrB);
            
            // With the adjusted pointers and Mat headers, run the single frame
            // segmentation.
            segment_singleframe_pipeline(ImgIn, sizex, sizey, 
                C, sizeC, A, B, BX, BY, Bth, Bdil, 
                NeuronXYCandidates, NeuronNCandidatesVolume[nu],
                maxX, maxY, maxXInStack, maxYInStack, threshold, blur, true);
            /**
            Move the pointers to the next frames in framesIn/ImgIn and 
            ArrBB/ArrB. While ArrB is resent to point to the beginning of ArrBB
            at each new volume, ImgIn just keeps going. For the time being, i
            is just to keep track of the current frame, but I'm not using it
            actually for anything.
            **/
            i++;
            ImgIn = ImgIn + sizexy*framesStride;
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

        }
    
        /**
        After the candidate neurons are found, select them using all the B 
        stored in ArrBB. 
        When iterating over nu = 0, 1, 2, framesN-1, framesN-2, framesN-3,
        pass the array Zero accordingly.
        **/
        
        int nu;
        uint32_t *NeuronXYin, *NeuronXYout;
        uint32_t NeuronNin;
        
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
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu]);

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
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu]);
            
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
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu]);
                
                NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            }
            
            // with B2 at nu=framesInVolume-2 (B4=Zero)
            nu = framesInVolume - 2;
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
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu]);
                    
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
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu]);
            
            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
        
        } else if(checkPlanesN==7) {
            // You need two additional pointers
            float *B5, *B6;
        
            // with B3 at nu=0 (B0=B1=B2=Zero)
            nu = 0;
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
                    
            segment_check2dcandidates_7planes(B0, B1, B2, B3, B4, B5, B6,
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu]);

            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            
            // with B3 at nu=1 (B0=B1=Zero)
            nu = 1;
            B0 = Zero;
            B1 = Zero;
            B2 = ArrBB;
            B3 = ArrBB + sizexy2;
            B4 = ArrBB + 2*sizexy2;
            B5 = ArrBB + 3*sizexy2;
            B6 = ArrBB + 4*sizexy2;
            NeuronXYin += NeuronNin;
            NeuronNin = NeuronNCandidatesVolume[1];
            NeuronXYout += NeuronNAll[volumeFirstFrame[mu]+nu-1];
            
            segment_check2dcandidates_7planes(B0, B1, B2, B3, B4, B5, B6,
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu]);
            
            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            
            // with B3 at nu=2 (B0=Zero)
            nu = 2;
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
            
            segment_check2dcandidates_7planes(B0, B1, B2, B3, B4, B5, B6,
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu]);
            
            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            
            for(int nu=3; nu<framesInVolume-3; nu++) {
                // Move the pointers to the right positions.
                
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
                
                segment_check2dcandidates_7planes(B0, B1, B2, B3, B4, B5, B6,
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu]);
                
                NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            }
            
            // with B3 at nu=framesInVolume-3 (B6=Zero)
            nu = framesInVolume - 3;
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
            
            segment_check2dcandidates_7planes(B0, B1, B2, B3, B4, B5, B6,
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu]);
                    
            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            
            // with B3 at nu=framesInVolume-2 (B6=B5=Zero)
            nu = framesInVolume - 2;
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
            
            segment_check2dcandidates_7planes(B0, B1, B2, B3, B4, B5, B6,
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu]);
                    
            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
            
            // with B3 at nu=framesInVolume-1 (B6=Zero)
            nu = framesInVolume - 1;
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
            
            segment_check2dcandidates_7planes(B0, B1, B2, B3, B4, B5, B6,
                    sizex2, sizey2, 
                    NeuronXYin, NeuronNin,
                    NeuronXYout, NeuronNAll[volumeFirstFrame[mu]+nu]);
                    
            NeuronNInAllPreviousVolumes += NeuronNAll[volumeFirstFrame[mu]+nu];
        }
    }
}

/*
TODO TRANSPARENT API AND GPU?
*/

void segment_singleframe_pipeline(uint16_t ImgIn[], 
	int32_t sizex, int32_t sizey, 
	cv::Mat& C, int32_t sizeC, 
	cv::Mat& A, cv::Mat& B, cv::Mat& BX, cv::Mat& BY, 
	cv::Mat& Bth, cv::Mat& Bdil,
	uint32_t NeuronXY[], uint32_t &NeuronN, 
	double &maxX, double &maxY, double maxXInStack, double maxYInStack,
	float threshold, double blur, bool resize) {
	
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
		cv::Mat K = cv::Mat(3, 3, CV_32F, cv::Scalar::all(1));
		
		// Other variables.
		double minX, minY, threshX, threshY;

		// Resize image
		cv::resize(cv::Mat(sizex, sizey, CV_16U, ImgIn), A, 
			A.size(), 0, 0, cv::INTER_AREA);

		// Apply Gaussian blur
		// 210 us
		//double blur = 0.65; //0.65
		cv::GaussianBlur(A, A, cv::Size(5, 5), blur, blur);

		// Calculate -d2/dx2 and -d2/dy2 with the filter passed as C.
		// In this way, the CV_32F images will saturate the positive 
		// derivatives and not the negative ones, which we need.
		// Before summing the two derivatives, they need to be divided by 2, 
		// so that they don't saturate. Instead of separately divide them,
		// use 1/2 instead of the 1 as second filter in the separable filter
		// function. (Saves 100 microseconds)
		// 300 us

		cv::sepFilter2D(A, BX, CV_32F, C, OneHalf);
		cv::sepFilter2D(A, BY, CV_32F, OneHalf, C);
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
		cv::minMaxIdx(BX, &minX, &maxX);
		cv::minMaxIdx(BY, &minY, &maxY);
		if ( (maxXInStack == -1.0 && maxYInStack == -1.0) || 
		     (maxXInStack < maxX || maxYInStack < maxY) ) {
			threshX = threshold * maxX;
			threshY = threshold * maxY;
		} else {
			threshX = threshold * maxXInStack;
			threshY = threshold * maxYInStack;
		}
		cv::threshold(BX, BX, threshX, 1, CV_THRESH_BINARY);
		cv::threshold(BY, BY, threshY, 1, CV_THRESH_BINARY);
		Bth = BX.mul(BY);
		Bth = Bth.mul(B);

		// Dilate
		// 60 us
		cv::dilate(Bth, Bdil, K);

		// Do a 1-dimensional scan over the arrays. Faster than two nested for
		// loops. You will have to calculate the xy coordinates from the single
		// index i.
		// 60 us
		float tmpBi;
		float tmpBdili;
		for (int i = 0; i < sizex2*sizey2; i++) {
			tmpBi = B.at<float>(i);
			tmpBdili = Bdil.at<float>(i);
			if (tmpBi != 0.0 && tmpBi == tmpBdili) {
				NeuronXY[k] = i;
				k++;
			}
		}
	
	}
	NeuronN = k;
}

void segment_check2dcandidates_5planes(
	float ArrB0[], float ArrB1[], float ArrB2[],
	float ArrB3[], float ArrB4[], int32_t sizeBx, int32_t sizeBy, 
	uint32_t NeuronXYin[], uint32_t NeuronNin,
	uint32_t *NeuronXYout, uint32_t &NeuronNout) {

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
**/		if (index > (uint32_t)sizeBx && index < upperlimit) {
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
	float ArrB0[], float ArrB1[], float ArrB2[],
	float ArrB3[], float ArrB4[], float ArrB5[], float ArrB6[],
    int32_t sizeBx, int32_t sizeBy, 
	uint32_t NeuronXYin[], uint32_t NeuronNin,
	uint32_t *NeuronXYout, uint32_t &NeuronNout) {

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
		Bxy = ArrB3[index];

		if (index > (uint32_t)sizeBx && index < upperlimit) {
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
		            (Bxy > ArrB5[index + sizeBx]) ) {
			    *(NeuronXYout+k) = index; 
			    k++;
		    }
		}
	}
	NeuronNout = k;
}

