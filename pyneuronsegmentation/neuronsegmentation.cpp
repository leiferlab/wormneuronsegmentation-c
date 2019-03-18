#include <stdint.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "neuronsegmentation.h"
#include <stdio.h>

void find_neurons_frames_sequence(uint16_t framesIn[],
    uint32_t framesN, int32_t sizex, int32_t sizey,
    int32_t framesStride, // 1 or 2 (RFP RFP RFP or RFP GFP RFP GFP)
    uint16_t ArrA[], 
    float ArrB[], float ArrBX[], float ArrBY[], 
	float ArrBth[], float ArrBdil[],
	uint32_t NeuronXY[], uint32_t NeuronN[]
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
    float threshold = 0.05;//0.25
            
    // Index for the frames with respect to the beginning
    ImgIn = framesIn;
    
    // For each volume
    maxXInStack = -1.0;
    maxYInStack = -1.0;
        
    // Run the single frame segmentation to initialize maxX/YInStack
    segment_singleframe_pipeline(ImgIn, sizex, sizey, 
        C, sizeC, A, B, BX, BY, Bth, Bdil, 
        NeuronXY, NeuronN[0],
        maxX, maxY, maxXInStack, maxYInStack, threshold, true);
        
    maxXInStack = maxX;
    maxYInStack = maxY;
        
    // Segment each frame independently and find the neurons.
    for(uint nu=0; nu<framesN; nu++) {
            maxXInStack = -1.0;
            maxYInStack = -1.0;
              
        // Run the single frame segmentation.
        segment_singleframe_pipeline(ImgIn, sizex, sizey, 
            C, sizeC, A, B, BX, BY, Bth, Bdil, 
            NeuronXY, NeuronN[nu],
            maxX, maxY, maxXInStack, maxYInStack, threshold, true);
        
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
            maxXInStack = (maxXInStack<maxX)?maxX:maxXInStack;
            maxYInStack = (maxYInStack<maxY)?maxY:maxYInStack;
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
	uint32_t NeuronXYAll[], uint32_t NeuronNAll[]
	) {
	
	/*
	NeuronXYCandidatesVolume and NeuronNCandidatesVolume will store the
	candidate neurons for the volume being processed. The candidates that are
	selected will be stored in NeuronXYAll and NeuronNAll. Therefore, the length
	of NeuronNAll will be the number of frames.
	*/
    
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
    //float *B0, *B1, *B2, *B3, *B4;
    
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
    
    /*
    The __All versions of these arrays contain the indexes and number of the
    candidates for all the planes in the volume. These pointers are the ones
    increased and passed to the segment_singleframe() function so that the
    latter can accumulate the results in the __All arrays.
    */
    uint32_t *NeuronXYCandidates, *NeuronNCandidates;
    
    double maxX, maxY, maxXInStack, maxYInStack;
    float threshold = 0.25;
    
    /** 
    You need to initialize maxXInStack and maxYInStack. Run the segmentation
    twice for the first volume.
    **/
    
    // TODO
    
    // Index for the frames with respect to the beginning
    int i = 0;
    ImgIn = framesIn;
    
    // For each volume
    for(uint mu=0; mu<volumeN; mu++) {
        ArrB = ArrBB;
        
        NeuronXYCandidates = NeuronXYCandidatesVolume;
        
        maxXInStack = 0.0;
        maxYInStack = 0.0;
        
        int framesInVolume = volumeFirstFrame[mu+1] - volumeFirstFrame[mu];
        
        // Segment each frame independently and find the candidate neurons.
        for(int nu=0; nu<framesInVolume; nu++) {
            
            // Create the Mat header for the pointer ArrB, that changes at every
            // iteration.
            B = cv::Mat(sizex2, sizey2, CV_32F, ArrB);
            
            // With the adjusted pointers and Mat headers, run the single frame
            // segmentation.
            segment_singleframe_pipeline(ImgIn, sizex, sizey, 
                C, sizeC, A, B, BX, BY, Bth, Bdil, 
                NeuronXYCandidates, NeuronNCandidatesVolume[nu],
                maxX, maxY, maxXInStack, maxYInStack, threshold, true);
            
            /**
            Move the pointers to the next frames in framesIn/ImgIn and 
            ArrBB/ArrB. While ArrB is resent to point to the beginning of ArrBB
            at each new volume, ImgIn just keeps going. For the time being, i
            is just to keep track of the current frame, but I'm not using it
            actually for anything.
            **/
            ImgIn = ImgIn + sizexy*framesStride;
            i++;
            ArrB = ArrB + sizexy2;
            
            /**
            Move NeuronXYCandidates further along NeuronXYCAndidatesVolume by 
            the number of candidates found in this volume, and 
            NeuronNCandidates to the next element of NeuronNCandidatesVolume.
            **/
            NeuronXYCandidates = NeuronXYCandidates + NeuronNCandidatesVolume[nu];
            NeuronNCandidates++;
            
            // Update encountered maxima.
            maxXInStack = (maxXInStack<maxX)?maxX:maxXInStack;
            maxYInStack = (maxYInStack<maxY)?maxY:maxYInStack;
        }
    
        /**
        After the candidate neurons are found, select them using all the B 
        stored in ArrBB. 
        When iterating over nu = 0,1,2 and nu = framesN-1, framesN-2, framesN-3,
        pass the array Zero accordingly.
        **/
        
        /**
        NeuronXYCandidates = 
        NeuronNCandidates = NeuronNCandidatesVolume;
        
        // with B2 at nu=0 (B0=B1=Zero)
        B0 = Zero;
        B1 = Zero;
        B2 = ArrBB;
        B3 = ArrBB + sizexy2;
        B4 = ArrBB + 2*sizexy2;
        NeuronXYin = NeuronXYCAndidatesVolume;
        NeuronNin = NeuronNCandidatesVolume[0];
        NeuronXYout = 
        
        segment_check2dcandidates_5planes(B0, B1, B2, B3, B4, 
                sizex2, sizey2, 
                NeuronXYin, NeuronNin,
                NeuronXYout, NeuronNout);
        
        // with B2 at nu=1 (B0=Zero)
        B0 = Zero;
        B1 = ArrBB;
        B2 = ArrBB + sizexy2;
        B3 = ArrBB + 2*sizexy2;
        B4 = ArrBB + 3*sizexy2;
        
        segment_check2dcandidates_5planes(B0, B1, B2, B3, B4, 
                sizex2, sizey2, 
                NeuronXYin, NeuronNin,
                NeuronXYout, NeuronNout);
        
        for(int nu=2; nu<framesInVolume-2; nu++) {
            
            // Move the pointers to the right positions.
            
            // The actual "Hessians"
            B0 = ArrBB + (nu-2)*sizexy2;
            B1 = ArrBB + (nu-1)*sizexy2;
            B2 = ArrBB + (nu)*sizexy2;
            B3 = ArrBB + (nu+1)*sizexy2;
            B4 = ArrBB + (nu+2)*sizexy2;
            
            // The indexes and number of the neuron candidates.
            
            NeuronXYin is a pointer to a position in NeuronXYCAndidatesVolume
            NeuronXYout is a pointer to a position in NeuronXYAll
            
            segment_check2dcandidates_5planes(B0, B1, B2, B3, B4, 
                sizex2, sizey2, 
                NeuronXYin, NeuronNin,
                NeuronXYout, NeuronNout);
        }
        
        // with B2 at nu=framesInVolume-2 (B4=Zero)
        B0 = ArrBB + (framesInVolume-4)*sizexy2;
        B1 = ArrBB + (framesInVolume-3)*sizexy2;
        B2 = ArrBB + (framesInVolume-2)*sizexy2;
        B3 = ArrBB + (framesInVolume-1)*sizexy2;
        B4 = Zero;
        
        segment_check2dcandidates_5planes(B0, B1, B2, B3, B4, 
                sizex2, sizey2, 
                NeuronXYin, NeuronNin,
                NeuronXYout, NeuronNout);
        
        // with B2 at nu=framesInVolume-1 (B4=B3=Zero)
        B0 = ArrBB + (framesInVolume-3)*sizexy2;
        B1 = ArrBB + (framesInVolume-2)*sizexy2;
        B2 = ArrBB + (framesInVolume-1)*sizexy2;
        B3 = Zero;
        B4 = Zero;
        
        segment_check2dcandidates_5planes(B0, B1, B2, B3, B4, 
                sizex2, sizey2, 
                NeuronXYin, NeuronNin,
                NeuronXYout, NeuronNout);
        **/
    }
}


/*
uint16_t *current_frame;

for on the volumes
    foreach frame
        current_frame = *(video_chunk + frame_index * 512 * 512)
        cv::Mat(sizex, sizey, CV_16U, current_frame)
TODO WHAT IS TRANSPARENT API?
*/

void segment_singleframe_pipeline(uint16_t ImgIn[], 
	int32_t sizex, int32_t sizey, 
	cv::Mat& C, int32_t sizeC, 
	cv::Mat& A, cv::Mat& B, cv::Mat& BX, cv::Mat& BY, 
	cv::Mat& Bth, cv::Mat& Bdil,
	uint32_t NeuronXY[], uint32_t &NeuronN, 
	double &maxX, double &maxY, double maxXInStack, double maxYInStack,
	float threshold, bool resize) {

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
		double blur = 0.65; //0.65
		cv::GaussianBlur(A, A, cv::Size(3, 3), blur, blur);

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
	uint32_t NeuronXYout[], uint32_t &NeuronNout) {

	// Brute force check if the candidate neurons found on each plane are
	// actually maxima in -d2/dx2-d2/dy2 also in a sphere around them and not
	// just in their plane.
	// This function can run as soon as the 5 planes are available, so that
	// you don't have to wait until the full volume is ready before starting
	// this brute force check.
	
	float Bxy;
	uint32_t k = 0;
	uint32_t index;
	bool ok;


	for (uint i = 0; i < NeuronNin; i++) {
		index = NeuronXYin[i];
		Bxy = ArrB2[index];

		ok = true;

		// I will hard code the regions where to look.
		// Since B is -d2/dx2-d2/dy2, I have to check if it's the maximum.
		
		///////////////////////////////
		///// - maybe do a for loop over the y-1,y,y+1, so that the compiler
		/////	can vectorize at least that.
		///////////////////////////////

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

		if (ok == true) {
			NeuronXYout[k] = index;
			k++;
		}
	}
	
	NeuronNout = k;
}

