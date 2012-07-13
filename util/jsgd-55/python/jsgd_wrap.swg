// -*- c++ -*- 

%module jsgd_wrap

/* most code here is borrowed from Yael 

   https://gforge.inria.fr/projects/jsgd/

*/

/*************************************************************************
 * functions release GIL by default
 */


%exception {
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}



/*************************************************************************
 * arrays
 */


%include <carrays.i>

/* 
   a=IntArray(4) constructs an array of 4 ints  
   a.cast() returns an int* usable as a function argument
   if x is a int* in C,  
   b=IntArray.frompointer(x) makes a[i] valid to access x[i]
   b=IntArray.acquirepointer(x) will also free(x) when b is deleted
   b.plus(2) returns a pointer to x+2
*/
%define array_class_plus(type,name)
%array_class(type,name)
%extend name {
  type *plus(long i) {
    return $self+i;
  }
  void clear(long n) {
    memset($self,0,sizeof(type)*n);
  }
  void copyfrom(type *src,long dest_ofs,long n) {
    memmove($self+dest_ofs,src,sizeof(type)*n);
  } 
  PyObject *tostring(long n) {
    PyObject *str = PyString_FromStringAndSize(NULL, n*sizeof(type));
    memcpy(PyString_AsString(str),$self,sizeof(type)*n);
    return str;
  }
  void fromstring(PyObject*obj) {
    if(!PyString_Check(obj)) {
      fprintf(stderr,"need string!");
      return;
    }
    memcpy($self,PyString_AsString(obj),PyString_Size(obj));
  }
};
%pythoncode %{
name.aptr=name.acquirepointer=staticmethod(lambda ptr: _frompointer_and_acquire(name,ptr))
%}
%enddef

%pythoncode %{
def _frompointer_and_acquire(aclass,ptr):
  r=aclass.frompointer(ptr)
  if r: r.this.acquire()
  return r
%}

array_class_plus(float,fvec)
array_class_plus(int,ivec)



/*************************************************************************
 * Numpy wrapper 
 */


%define NUMPY_EXCHANGE(xvec, ctype, numpytype)

%{

// declare in C

PyObject * xvec ## _to_numpy(const ctype *src,long di) {
  npy_intp d = di;
  PyObject* ob = PyArray_SimpleNew(1, &d, numpytype);
  ctype *dest = PyArray_DATA(ob);
  memcpy(dest, src, sizeof(ctype) * d);
  return ob;
}

PyObject * xvec ## _to_numpy_ref(ctype *src, long di) {
  npy_intp d = di;
  return PyArray_SimpleNewFromData(1, &d, numpytype, src);
}


ctype *numpy_to_ ## xvec(PyObject *a) {
  CHECK_NUMPY_ARRAY(numpytype)
  size_t sz = PyArray_SIZE(a) * PyArray_ITEMSIZE(a);
  void *ret = malloc(sz);
  memcpy(ret, PyArray_DATA(a), sz);
  return ret;
}

ctype *numpy_to_ ## xvec ## _ref(PyObject *a) {
  CHECK_NUMPY_ARRAY(numpytype)
  return PyArray_DATA(a);
}

%}

// re-declare for SWIG

PyObject * xvec ## _to_numpy(const ctype *src,long di);
PyObject * xvec ## _to_numpy_ref(ctype *src, long di);
ctype *numpy_to_ ## xvec(PyObject *a);
ctype *numpy_to_ ## xvec ## _ref(PyObject *a);

%enddef






%{
#define SWIG_FILE_WITH_INIT

#include <numpy/arrayobject.h>

#define CHECK_NUMPY_ARRAY(expected_npy)                                 \
  if(!a) {                                                              \
    fprintf(stderr,"%s: NULL input\n",__PRETTY_FUNCTION__);             \
    return NULL;                                                        \
  }                                                                     \
  if(!PyArray_Check(a)) {                                               \
    fprintf(stderr,"%s: input not numpy array\n",__PRETTY_FUNCTION__);  \
    return NULL;                                                        \
  }                                                                     \
  if(PyArray_TYPE(a)!=expected_npy) {                                   \
    fprintf(stderr,"%s: input has bad type (type id %d != expected %d)\n",__PRETTY_FUNCTION__, \
            PyArray_TYPE(a),expected_npy);                              \
    return NULL;                                                        \
  }

%}


%init %{
/* needed, else crash at runtime */
    import_array();
%}



NUMPY_EXCHANGE(bvec, unsigned char, NPY_UINT8);
NUMPY_EXCHANGE(fvec, float, NPY_FLOAT32);
NUMPY_EXCHANGE(ivec, int, NPY_INT32);
NUMPY_EXCHANGE(dvec, double, NPY_FLOAT64);

/*************************************************************************
 * jsgd
 */


%{

#include "../c/jsgd.h"


%}

%include "../c/x_matrix.h"
%include "../c/jsgd.h"

