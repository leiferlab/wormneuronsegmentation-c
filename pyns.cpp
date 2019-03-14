//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <stdint.h>
#include "neuronsegmentation.cpp"

static PyObject *pyns_test(PyObject *self, PyObject *args);
static PyObject *pyns_find_neurons(PyObject *self, PyObject *args);
static PyObject *pyns_find_neurons_frames_sequence(PyObject *self, PyObject *args);

/////// Python-module-related functions and tables

// The module's method table
static PyMethodDef pynsMethods[] = {
    /*{"initRegistration", pygmmreg_initRegistration, METH_VARARGS,
        "Initialize the registration (regTransformationCostFunction class)"},*/
    {"test", pyns_test, METH_VARARGS, ""},
    {"find_neurons", pyns_find_neurons, METH_VARARGS, ""},
    {"find_neurons_frames_sequence", pyns_find_neurons_frames_sequence, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

// The module definition function
static struct PyModuleDef pyns = {
    PyModuleDef_HEAD_INIT,
    "pyns",
    NULL, // Module documentation
    -1,
    pynsMethods
};

// The module initialization function
PyMODINIT_FUNC PyInit_pyns(void) { 
        import_array(); //Numpy
        return PyModule_Create(&pyns);
    }
    
void funzione(uint16_t c[]) { c[2] = 1; }
    
//////// The actual functions of the modules

static PyObject *pyns_find_neurons_frames_sequence(PyObject *self, PyObject *args) {

    int framesN;
    PyObject *framesIn_o;
    
    int sizex, sizey, framesStride;
    
    PyObject *ArrA_o, *ArrB_o, *ArrBX_o, *ArrBY_o, *ArrBth_o, *ArrBdil_o;
    PyObject *NeuronXY_o, *NeuronN_o;
    
    if(!PyArg_ParseTuple(args, "iOiiiOOOOOOOO", 
            &framesN, &framesIn_o, &sizex, &sizey, &framesStride,
            &ArrA_o, &ArrB_o, &ArrBX_o, &ArrBY_o, &ArrBth_o, &ArrBdil_o,
            &NeuronXY_o, &NeuronN_o)) return NULL;
    
    PyObject *framesIn_a = PyArray_FROM_OTF(framesIn_o, NPY_UINT16, NPY_IN_ARRAY);
    PyObject *ArrA_a = PyArray_FROM_OTF(ArrA_o, NPY_UINT16, NPY_IN_ARRAY);
    PyObject *ArrB_a = PyArray_FROM_OTF(ArrB_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *ArrBX_a = PyArray_FROM_OTF(ArrBX_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *ArrBY_a = PyArray_FROM_OTF(ArrBY_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *ArrBth_a = PyArray_FROM_OTF(ArrBth_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *ArrBdil_a = PyArray_FROM_OTF(ArrBdil_o, NPY_FLOAT32, NPY_IN_ARRAY);
    PyObject *NeuronXY_a = PyArray_FROM_OTF(NeuronXY_o, NPY_UINT32, NPY_IN_ARRAY);
    PyObject *NeuronN_a = PyArray_FROM_OTF(NeuronN_o, NPY_UINT32, NPY_IN_ARRAY);
    
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
        NeuronN_a == NULL
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
    
    //////////////////////////////////
    //////////////////////////////////
    // Actual C code
    //////////////////////////////////
    //////////////////////////////////
    
    find_neurons_frames_sequence(framesIn, framesN, sizex, sizey,
        framesStride, // 1 or 2 (RFP RFP RFP or RFP GFP RFP GFP)
        ArrA, ArrB, ArrBX, ArrBY, ArrBth, ArrBdil, 
	    NeuronXY, NeuronN);
    
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
    
    // Return the python object none. Its reference count has to be increased.
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *pyns_find_neurons(PyObject *self, PyObject *args) {

    int framesN;
    PyObject *framesIn_o;
    
    int sizex, sizey, framesStride;
    
    int volumeN;
    PyObject *volumeFirstFrame_o;
    
    PyObject *ArrA_o, *ArrBB_o, *ArrBX_o, *ArrBY_o, *ArrBth_o, *ArrBdil_o;
    PyObject *NeuronXYCandidatesVolume_o, *NeuronNCandidatesVolume_o;
    PyObject *NeuronXYAll_o, *NeuronNAll_o;
    
    
    if(!PyArg_ParseTuple(args, "iOiiiiOOOOOOOOOOO", 
            &framesN, &framesIn_o, &sizex, &sizey, &framesStride,
            &volumeN, &volumeFirstFrame_o,
            &ArrA_o, &ArrBB_o, &ArrBX_o, &ArrBY_o, &ArrBth_o, &ArrBdil_o,
            &NeuronXYCandidatesVolume_o, &NeuronNCandidatesVolume_o,
            &NeuronXYAll_o, &NeuronNAll_o)) return NULL;
    
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
        NeuronNAll_a == NULL
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
        return NULL;
    }
    
    // Get pointers to the data in the numpy arrays.
    uint16_t *framesIn = (uint16_t*)PyArray_DATA(framesIn_a);
    uint32_t *volumeFirstFrame = (uint32_t*)PyArray_DATA(volumeFirstFrame);
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
    
    //////////////////////////////////
    //////////////////////////////////
    // Actual C code
    //////////////////////////////////
    //////////////////////////////////
    
    find_neurons(framesIn, framesN, sizex, sizey,
        framesStride, // 1 or 2 (RFP RFP RFP or RFP GFP RFP GFP)
        volumeFirstFrame, volumeN,
        ArrA, ArrBB, ArrBX, ArrBY, ArrBth, ArrBdil, 
        NeuronXYCandidatesVolume, 
	    NeuronNCandidatesVolume,
	    NeuronXYAll, NeuronNAll);
    
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
    
    // Return the python object none. Its reference count has to be increased.
    Py_INCREF(Py_None);
    return Py_None;
}



static PyObject *pyns_test(PyObject *self, PyObject *args) {

    int nprova;
    PyObject *prova_obj;
    
    if(!PyArg_ParseTuple(args, "iO", &nprova, &prova_obj)) return NULL;
    
    PyObject *prova_arr = PyArray_FROM_OTF(prova_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    
    // Check that the above conversion worked, otherwise decrease the reference
    // count and return NULL.                                 
    if (prova_arr == NULL ) {
        Py_XDECREF(prova_arr);
        return NULL;
    }
    
    // Get pointers to the data in the numpy arrays.
    double *prova = (double*)PyArray_DATA(prova_arr);
    
    //////////////////////////////////
    //////////////////////////////////
    // Actual C code
    //////////////////////////////////
    //////////////////////////////////
    
    uint16_t c[3] = {0,0,0};
    funzione(c);
    std::cout << c[2];
    
    uint16_t a[3] = {0,0,0};
    uint16_t *b;
    
    b = a ;
    funzione(b);
    //*(b) = 3;
    
    std::cout << a[0];
    std::cout << a[1];
    std::cout << a[2];
    
    //////////////////////////////////
    //////////////////////////////////
    // End of C code
    //////////////////////////////////
    //////////////////////////////////
    
    
    // Decrease the reference count for the python objects that have been 
    // declared in this function.
    Py_XDECREF(prova_arr);
    
    // Return the python object none. Its reference count has to be increased.
    Py_INCREF(Py_None);
    return Py_None;
}

