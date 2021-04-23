/**
    wormneuronsegmentation
    _wormneuronsegmentation_c.cpp
    Finds nuclei of neurons in stack of fluorescence images.

    @author Francesco Randi
**/

//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <stdint.h>
#include "../neuronsegmentation_c/neuronsegmentation.hpp"

static PyObject *wormns_find_neurons(PyObject *self, PyObject *args);
static PyObject *wormns_find_neurons_frames_sequence(PyObject *self, PyObject *args);

/////// Python-module-related functions and tables

// The module's method table
static PyMethodDef _wormneuronsegmentation_cMethods[] = {
    {"find_neurons", wormns_find_neurons, METH_VARARGS, ""},
    {"find_neurons_frames_sequence", wormns_find_neurons_frames_sequence, METH_VARARGS, 
        "\
        Parameters \n\
        ---------- \n\
        framesN: np.uint32 \n\
            Total number of images (frames) passed in framesIn. \n\
        framesIn: np.array of np.uint16 \n\
            Array containing the images to be analyzed as [index, x, y]. Can \n\
            contain multiple channels as [index, channel, x, y] \n\
            (see framesStride). \n\
        sizex: np.uint32 \n\
            Size along x of the images. \n\
        sizey: np.uint32 \n\
            Size along y of the images. \n\
        framesStride: np.uint32 \n\
            Number of channels present for each image (e.g. 2 for RFP GFP). \n\
        ArrA: np.array of np.float32 \n\
            Allocated array used in the analysis (and diagnostics). \n\
            Must have half the size of the frames in each dimensions (e.g. if\n\
            framesIn has shape (_,_,512,512), this array must be (256,256). \n\
        ArrB: np.array of np.float32 \n\
            Allocated array used in the analysis (and diagnostics). \n\
            Must have half the size of the frames in each dimensions (e.g. if\n\
            framesIn has shape (_,_,512,512), this array must be (256,256).\n\
        ArrBX: np.array of np.float32 \n\
            Allocated array used in the analysis (and diagnostics). \n\
            Must have half the size of the frames in each dimensions (e.g. if\n\
            framesIn has shape (_,_,512,512), this array must be (256,256).\n\
        ArrBY: np.array of np.float32 \n\
            Allocated array used in the analysis (and diagnostics). \n\
            Must have half the size of the frames in each dimensions (e.g. if\n\
            framesIn has shape (_,_,512,512), this array must be (256,256).\n\
        ArrBth: np.array of np.float32 \n\
            Allocated array used in the analysis (and diagnostics). \n\
            Must have half the size of the frames in each dimensions (e.g. if\n\
            framesIn has shape (_,_,512,512), this array must be (256,256). \n\
        NeuronXY: np.array of np.uint32 \n\
            Array that will be populated with the coordinates of the neurons\n\
            found (represented as 1D coordinates in the linearized image).\n\
            The coordinates of the neurons found in the different frames will\n\
            be stored in this array sequentially, and can be split using the\n\
            array NeuronN. \n\
            This array needs to have more elements than the total number of \n\
            neurons that will be found, otherwise a Segmentation fault will \n\
            occur. Make safe estimates: this array will take anyway a small \n\
            amount of memory. \n\
        NeuronN: np.array of np.uint32 \n\
            Array that will be populated with the number of neurons found in \n\
            each frame (NeuronN[i] is the number of neurons found in frame i).\n\
            Its size must be the number of frames passed.\n\
        \n\
        Returns\n\
        -------\n\
        None. \n\
        \n\
        Finds the position of the neurons in a sequence of images not \n\
        belonging to volumes, i.e. the neurons are found in 2D and there is \n\
        no 3D selection of the candidate neurons, as in the function \n\
        find_neurons(). \n\
        "},
    {NULL, NULL, 0, NULL}
};

// The module definition function
static struct PyModuleDef _wormneuronsegmentation_c = {
    PyModuleDef_HEAD_INIT,
    "_wormneuronsegmentation_c",
    NULL, // Module documentation
    -1,
    _wormneuronsegmentation_cMethods
};

// The module initialization function
PyMODINIT_FUNC PyInit__wormneuronsegmentation_c(void) { 
        import_array(); //Numpy
        return PyModule_Create(&_wormneuronsegmentation_c);
    }
    
    
//////// The actual functions of the modules

static PyObject *wormns_find_neurons_frames_sequence(PyObject *self, PyObject *args) {

    int framesN;
    PyObject *framesIn_o;
    
    int sizex, sizey;
    int32_t framesStride;
    uint32_t extractCurvatureBoxSize;
    uint32_t dil_size;
    float threshold;
    double blur;
    
    PyObject *ArrA_o, *ArrB_o, *ArrBX_o, *ArrBY_o, *ArrBth_o, *ArrBdil_o;
    PyObject *NeuronXY_o, *NeuronN_o,*NeuronCurvature_o;
    
    if(!PyArg_ParseTuple(args, "iOiiiOOOOOOOOOfdii", 
            &framesN, &framesIn_o, &sizex, &sizey, &framesStride,
            &ArrA_o, &ArrB_o, &ArrBX_o, &ArrBY_o, &ArrBth_o, &ArrBdil_o,
            &NeuronXY_o, &NeuronN_o, &NeuronCurvature_o,
            &threshold, &blur, &dil_size,
            &extractCurvatureBoxSize)) return NULL;
    
    PyObject *framesIn_a = PyArray_FROM_OTF(framesIn_o, NPY_UINT16, NPY_IN_ARRAY);
    PyObject *ArrA_a = PyArray_FROM_OTF(ArrA_o, NPY_UINT16, NPY_IN_ARRAY);
    PyObject *ArrB_a = PyArray_FROM_OTF(ArrB_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *ArrBX_a = PyArray_FROM_OTF(ArrBX_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *ArrBY_a = PyArray_FROM_OTF(ArrBY_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *ArrBth_a = PyArray_FROM_OTF(ArrBth_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *ArrBdil_a = PyArray_FROM_OTF(ArrBdil_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *NeuronXY_a = PyArray_FROM_OTF(NeuronXY_o, NPY_UINT32, NPY_IN_ARRAY);
    PyObject *NeuronN_a = PyArray_FROM_OTF(NeuronN_o, NPY_UINT32, NPY_IN_ARRAY);
    PyObject *NeuronCurvature_a = PyArray_FROM_OTF(NeuronCurvature_o, NPY_FLOAT32, NPY_IN_ARRAY);
    
    // Check that the above conversion worked, otherwise decrease the reference
    // count and return NULL.                                 
    if (framesIn_a == NULL ||
        ArrA_a == NULL ||
        ArrB_a == NULL ||
        ArrBX_a == NULL ||
        ArrBY_a == NULL ||
        ArrBth_a == NULL ||
        ArrBdil_a == NULL ||
        NeuronXY_a == NULL ||
        NeuronN_a == NULL ||
        NeuronCurvature_a == NULL
        ) {
        Py_XDECREF(framesIn_a);
        Py_XDECREF(ArrA_a);
        Py_XDECREF(ArrB_a);
        Py_XDECREF(ArrBX_a);
        Py_XDECREF(ArrBY_a);
        Py_XDECREF(ArrBth_a);
        Py_XDECREF(ArrBdil_a);
        Py_XDECREF(NeuronXY_a);
        Py_XDECREF(NeuronN_a);
        Py_XDECREF(NeuronCurvature_a);
        return NULL;
    }
    
    // Get pointers to the data in the numpy arrays.
    uint16_t *framesIn = (uint16_t*)PyArray_DATA(framesIn_a);
    uint16_t *ArrA = (uint16_t*)PyArray_DATA(ArrA_a);
    float *ArrB = (float*)PyArray_DATA(ArrB_a);
    float *ArrBX = (float*)PyArray_DATA(ArrBX_a);
    float *ArrBY = (float*)PyArray_DATA(ArrBY_a);
    float *ArrBth = (float*)PyArray_DATA(ArrBth_a);
    float *ArrBdil = (float*)PyArray_DATA(ArrBdil_a);
    uint32_t *NeuronXY = (uint32_t*)PyArray_DATA(NeuronXY_a);
    uint32_t *NeuronN = (uint32_t*)PyArray_DATA(NeuronN_a);
    float *NeuronCurvature = (float*)PyArray_DATA(NeuronCurvature_a);
    
    //////////////////////////////////
    //////////////////////////////////
    // Actual C code
    //////////////////////////////////
    //////////////////////////////////
    
    find_neurons_frames_sequence(framesIn, framesN, sizex, sizey,
        framesStride, // 1 or 2 (RFP RFP RFP or RFP GFP RFP GFP)
        ArrA, ArrB, ArrBX, ArrBY, ArrBth, ArrBdil, 
	    NeuronXY, NeuronN, NeuronCurvature,
	    threshold, blur, dil_size, extractCurvatureBoxSize);
    
    //////////////////////////////////
    //////////////////////////////////
    // End of C code
    //////////////////////////////////
    //////////////////////////////////
    
    
    // Decrease the reference count for the python objects that have been 
    // declared in this function.
    Py_XDECREF(framesIn_a);
    Py_XDECREF(ArrA_a);
    Py_XDECREF(ArrB_a);
    Py_XDECREF(ArrBX_a);
    Py_XDECREF(ArrBY_a);
    Py_XDECREF(ArrBth_a);
    Py_XDECREF(ArrBdil_a);
    Py_XDECREF(NeuronXY_a);
    Py_XDECREF(NeuronN_a);
    Py_XDECREF(NeuronCurvature_a);
    
    // Return the python object none. Its reference count has to be increased.
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *wormns_find_neurons(PyObject *self, PyObject *args) {

    int framesN;
    PyObject *framesIn_o;
    
    int sizex, sizey;
    int32_t frame0, framesStride;
    
    int volumeN;
    PyObject *volumeFirstFrame_o;
    
    float threshold;
    double blur;
    uint32_t checkPlanesN;
    uint32_t xydiameter, dil_size;
    uint32_t extractCurvatureBoxSize;
    int candidateCheck_i;
    bool candidateCheck = true;
    
    PyObject *ArrA_o, *ArrBB_o, *ArrBX_o, *ArrBY_o, *ArrBth_o, *ArrBdil_o;
    PyObject *NeuronXYCandidatesVolume_o, *NeuronNCandidatesVolume_o;
    PyObject *NeuronXYAll_o, *NeuronNAll_o;
    PyObject *NeuronCurvatureAll_o;
    
    
    if(!PyArg_ParseTuple(args, "iOiiiiiOOOOOOOOOOOOfdiiiii", 
            &framesN, &framesIn_o, &sizex, &sizey, &frame0, &framesStride,
            &volumeN, &volumeFirstFrame_o,
            &ArrA_o, &ArrBB_o, &ArrBX_o, &ArrBY_o, &ArrBth_o, &ArrBdil_o,
            &NeuronXYCandidatesVolume_o, &NeuronNCandidatesVolume_o,
            &NeuronXYAll_o, &NeuronNAll_o,
            &NeuronCurvatureAll_o,
            &threshold, &blur, &dil_size, &checkPlanesN, &xydiameter,
            &extractCurvatureBoxSize, &candidateCheck_i)) return NULL;
    
    PyObject *framesIn_a = PyArray_FROM_OTF(framesIn_o, NPY_UINT16, NPY_IN_ARRAY);
    PyObject *volumeFirstFrame_a = PyArray_FROM_OTF(volumeFirstFrame_o, NPY_UINT32, NPY_IN_ARRAY);
    PyObject *ArrA_a = PyArray_FROM_OTF(ArrA_o, NPY_UINT16, NPY_IN_ARRAY);
    PyObject *ArrBB_a = PyArray_FROM_OTF(ArrBB_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *ArrBX_a = PyArray_FROM_OTF(ArrBX_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *ArrBY_a = PyArray_FROM_OTF(ArrBY_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *ArrBth_a = PyArray_FROM_OTF(ArrBth_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *ArrBdil_a = PyArray_FROM_OTF(ArrBdil_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *NeuronXYCandidatesVolume_a = PyArray_FROM_OTF(NeuronXYCandidatesVolume_o, NPY_UINT32, NPY_IN_ARRAY);
    PyObject *NeuronNCandidatesVolume_a = PyArray_FROM_OTF(NeuronNCandidatesVolume_o, NPY_UINT32, NPY_IN_ARRAY);
    PyObject *NeuronXYAll_a = PyArray_FROM_OTF(NeuronXYAll_o, NPY_UINT32, NPY_IN_ARRAY);
    PyObject *NeuronNAll_a = PyArray_FROM_OTF(NeuronNAll_o, NPY_UINT32, NPY_IN_ARRAY);
    PyObject *NeuronCurvatureAll_a = PyArray_FROM_OTF(NeuronCurvatureAll_o, NPY_FLOAT32, NPY_IN_ARRAY);
    
    // Check that the above conversion worked, otherwise decrease the reference
    // count and return NULL.                                 
    if (framesIn_a == NULL ||
        volumeFirstFrame_a == NULL ||
        ArrA_a == NULL ||
        ArrBB_a == NULL ||
        ArrBX_a == NULL ||
        ArrBY_a == NULL ||
        ArrBth_a == NULL ||
        ArrBdil_a == NULL ||
        NeuronXYCandidatesVolume_a == NULL ||
        NeuronNCandidatesVolume_a == NULL ||
        NeuronXYAll_a == NULL ||
        NeuronNAll_a == NULL ||
        NeuronCurvatureAll_a == NULL
        ) {
        Py_XDECREF(framesIn_a);
        Py_XDECREF(volumeFirstFrame_a);
        Py_XDECREF(ArrA_a);
        Py_XDECREF(ArrBB_a);
        Py_XDECREF(ArrBX_a);
        Py_XDECREF(ArrBY_a);
        Py_XDECREF(ArrBth_a);
        Py_XDECREF(ArrBdil_a);
        Py_XDECREF(NeuronXYCandidatesVolume_a);
        Py_XDECREF(NeuronNCandidatesVolume_a);
        Py_XDECREF(NeuronXYAll_a);
        Py_XDECREF(NeuronNAll_a);
        Py_XDECREF(NeuronCurvatureAll_a);
        return NULL;
    }
    
    // Get pointers to the data in the numpy arrays.
    uint16_t *framesIn = (uint16_t*)PyArray_DATA(framesIn_a);
    uint32_t *volumeFirstFrame = (uint32_t*)PyArray_DATA(volumeFirstFrame_a);
    uint16_t *ArrA = (uint16_t*)PyArray_DATA(ArrA_a);
    float *ArrBB = (float*)PyArray_DATA(ArrBB_a);
    float *ArrBX = (float*)PyArray_DATA(ArrBX_a);
    float *ArrBY = (float*)PyArray_DATA(ArrBY_a);
    float *ArrBth = (float*)PyArray_DATA(ArrBth_a);
    float *ArrBdil = (float*)PyArray_DATA(ArrBdil_a);
    uint32_t *NeuronXYCandidatesVolume = (uint32_t*)PyArray_DATA(NeuronXYCandidatesVolume_a);
    uint32_t *NeuronNCandidatesVolume = (uint32_t*)PyArray_DATA(NeuronNCandidatesVolume_a);
    uint32_t *NeuronXYAll = (uint32_t*)PyArray_DATA(NeuronXYAll_a);
    uint32_t *NeuronNAll = (uint32_t*)PyArray_DATA(NeuronNAll_a);
    float *NeuronCurvatureAll = (float*)PyArray_DATA(NeuronCurvatureAll_a);
    
    if(candidateCheck_i==0){candidateCheck=false;}
    
    //////////////////////////////////
    //////////////////////////////////
    // Actual C code
    //////////////////////////////////
    //////////////////////////////////
    
    find_neurons(framesIn, framesN, sizex, sizey,
        frame0,
        framesStride, // 1 or 2 (RFP RFP RFP or RFP GFP RFP GFP)
        volumeFirstFrame, volumeN,
        ArrA, ArrBB, ArrBX, ArrBY, ArrBth, ArrBdil, 
        NeuronXYCandidatesVolume, 
	    NeuronNCandidatesVolume,
	    NeuronXYAll, NeuronNAll,
	    NeuronCurvatureAll,
	    threshold, blur, dil_size,
	    checkPlanesN, xydiameter, extractCurvatureBoxSize,
	    candidateCheck);
    
    //////////////////////////////////
    //////////////////////////////////
    // End of C code
    //////////////////////////////////
    //////////////////////////////////
    
    
    // Decrease the reference count for the python objects that have been 
    // declared in this function.
    Py_XDECREF(framesIn_a);
    Py_XDECREF(volumeFirstFrame_a);
    Py_XDECREF(ArrA_a);
    Py_XDECREF(ArrBB_a);
    Py_XDECREF(ArrBX_a);
    Py_XDECREF(ArrBY_a);
    Py_XDECREF(ArrBth_a);
    Py_XDECREF(ArrBdil_a);
    Py_XDECREF(NeuronXYCandidatesVolume_a);
    Py_XDECREF(NeuronNCandidatesVolume_a);
    Py_XDECREF(NeuronXYAll_a);
    Py_XDECREF(NeuronNAll_a);
    Py_XDECREF(NeuronCurvatureAll_a);
    
    // Return the python object none. Its reference count has to be increased.
    Py_INCREF(Py_None);
    return Py_None;
}

