/**
    wormneuronsegmentation
    neuronsegmentation.h
    Finds nuclei of neurons in stack of fluorescence images.

    @author Francesco Randi
**/

#include <opencv2/core.hpp>


void segment_singleframe_pipeline(uint16_t ImgIn[], 
	int32_t sizex, int32_t sizey, 
	cv::Mat& C, int32_t sizeC, 
	cv::Mat& A, cv::Mat& B, cv::Mat& BX, cv::Mat& BY, 
	cv::Mat& Bth, cv::Mat& Bdil, cv::Mat& K,
	uint32_t NeuronXY[], uint32_t &NeuronN, 
	double &maxX, double &maxY, double maxXInStack, double maxYInStack,
	int &chunk_tot_count, int chunk_tot_thresh=10,
	float threshold = 0.25, double blur = 0.65, uint16_t A_thresh = 110,
	bool resize = true);

void segment_check2dcandidates_5planes(
	float ArrB0[], float ArrB1[], float ArrB2[],
	float ArrB3[], float ArrB4[], int32_t sizeBx, int32_t sizeBy, 
	uint32_t NeuronXYin[], uint32_t NeuronNin,
	uint32_t *NeuronXYout, uint32_t &NeuronNout,
	int32_t maxNeuronNPerVolume=400);

// This is an alias for either of the two versions (3 or 5 max diameter)
void segment_check2dcandidates_7planes(
    uint16_t ArrA0[], uint16_t ArrA1[], uint16_t ArrA2[],
    uint16_t ArrA3[], uint16_t ArrA4[], uint16_t ArrA5[], uint16_t ArrA6[],
	float ArrB0[], float ArrB1[], float ArrB2[],
	float ArrB3[], float ArrB4[], float ArrB5[], float ArrB6[],
	int32_t sizeBx, int32_t sizeBy, 
	uint32_t NeuronXYin[], uint32_t NeuronNin,
	uint32_t *NeuronXYout, uint32_t &NeuronNout, uint32_t maxdiameter,
	int32_t maxNeuronNPerVolume=400);

void segment_check2dcandidates_7planes_3maxdiameter(
    uint16_t ArrA0[], uint16_t ArrA1[], uint16_t ArrA2[],
    uint16_t ArrA3[], uint16_t ArrA4[], uint16_t ArrA5[], uint16_t ArrA6[],
	float ArrB0[], float ArrB1[], float ArrB2[],
	float ArrB3[], float ArrB4[], float ArrB5[], float ArrB6[],
	int32_t sizeBx, int32_t sizeBy, 
	uint32_t NeuronXYin[], uint32_t NeuronNin,
	uint32_t *NeuronXYout, uint32_t &NeuronNout,
	int32_t maxNeuronNPerVolume=400);
	
void segment_check2dcandidates_7planes_5maxdiameter(
    uint16_t ArrA0[], uint16_t ArrA1[], uint16_t ArrA2[],
    uint16_t ArrA3[], uint16_t ArrA4[], uint16_t ArrA5[], uint16_t ArrA6[],
	float ArrB0[], float ArrB1[], float ArrB2[],
	float ArrB3[], float ArrB4[], float ArrB5[], float ArrB6[],
	int32_t sizeBx, int32_t sizeBy, 
	uint32_t NeuronXYin[], uint32_t NeuronNin,
	uint32_t *NeuronXYout, uint32_t &NeuronNout,
	int32_t maxNeuronNPerVolume=400);
	
void segment_extract_curvature(
	float ArrB0[], float ArrB1[], float ArrB2[],
	float ArrB3[], float ArrB4[], float ArrB5[], float ArrB6[],
    int32_t sizeBx, int32_t sizeBy, 
	uint32_t NeuronXYin[], uint32_t NeuronNin,
	float *NeuronCurvatureOut, uint32_t totalBoxSize=51);

void segment_extract_curvature_single_frame(
	float ArrB[],
    int32_t sizeBx, int32_t sizeBy, 
	uint32_t NeuronXYin[], uint32_t NeuronNin,
	float *NeuronCurvatureOut, uint32_t totalBoxSize=13);
	
void find_neurons_frames_sequence_c(uint16_t framesIn[],
    uint32_t framesN, int32_t sizex, int32_t sizey,
    int32_t framesStride,
    uint16_t ArrA[], 
    float ArrB[], float ArrBX[], float ArrBY[], 
	float ArrBth[], float ArrBdil[],
	uint32_t NeuronXY[], uint32_t NeuronN[],
	float NeuronCurvature[],
	float threshold = 0.25, double blur = 0.65, uint32_t dil_size=7,
	uint16_t A_thresh=110,
	uint32_t extractCurvatureBoxSize=13);
	
void find_neurons(uint16_t framesIn[],
    uint32_t framesN, int32_t sizex, int32_t sizey,
    int32_t frame0,
    int32_t framesStride,
    uint32_t volumeFirstFrame[], uint32_t volumeN,
    uint16_t ArrA[], 
    float ArrBB[], float ArrBX[], float ArrBY[], 
	float ArrBth[], float ArrBdil[],
	uint32_t NeuronXYCandidatesVolume[], 
	uint32_t NeuronNCandidatesVolume[],
	uint32_t NeuronXYAll[], uint32_t NeuronNAll[],
	float NeuronCurvatureAll[],
	float threshold = 0.25, double blur = 0.65, uint32_t dil_size=7, 
	uint16_t A_thresh=110,
	uint32_t checkPlanesN = 5,
	uint32_t xydiameter=3, uint32_t extractCurvatureBoxSize=51,
	bool candidateCheck=true,
	int32_t maxNeuronNPerVolume=400);
	
float kthlargest(float array[], int size, int k);
