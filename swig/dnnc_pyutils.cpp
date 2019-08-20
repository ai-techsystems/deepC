#include <Python.h> // Must be first
#include <vector>
#include <stdexcept>

using namespace std;

// =====
// LISTS
// =====

PyObject* vectorToList_Float(const vector<float> &data) {
  PyObject* listObj = PyList_New( data.size() );
	if (!listObj) throw logic_error("Unable to allocate memory for Python list");
	for (unsigned int i = 0; i < data.size(); i++) {
		PyObject *num = PyFloat_FromDouble( (double) data[i]);
		if (!num) {
			Py_DECREF(listObj);
            throw std::bad_alloc();
		}
		PyList_SET_ITEM(listObj, i, num);
	}
	return listObj;
}

// ======
// TUPLES
// ======

PyObject* vectorToTuple_Float(const vector<float> &data) {
	PyObject* tuple = PyTuple_New( data.size() );
	if (!tuple) throw std::bad_alloc();
	for (unsigned int i = 0; i < data.size(); i++) {
		PyObject *num = PyFloat_FromDouble( (double) data[i]);
		if (!num) {
			Py_DECREF(tuple);
            throw std::bad_alloc();
		}
		PyTuple_SET_ITEM(tuple, i, num);
	}

	return tuple;
}

PyObject* vectorVectorToTuple_Float(const vector< vector< float > > &data) {
	PyObject* tuple = PyTuple_New( data.size() );
	if (!tuple) throw std::bad_alloc();
	for (unsigned int i = 0; i < data.size(); i++) {
		PyObject* subTuple = NULL;
		try {
			subTuple = vectorToTuple_Float(data[i]);
		} catch (logic_error &e) {
			throw e;
		}
		if (!subTuple) {
			Py_DECREF(tuple);
            throw std::bad_alloc();
		}
		PyTuple_SET_ITEM(tuple, i, subTuple);
	}

	return tuple;
}

// PyObject -> Vector
vector<float> listTupleToVector_Float(PyObject* incoming) {
	vector<float> data;
	if (PyTuple_Check(incoming)) {
		for(Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
			PyObject *value = PyTuple_GetItem(incoming, i);
			data.push_back( PyFloat_AsDouble(value) );
		}
	} else {
		if (PyList_Check(incoming)) {
			for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
				PyObject *value = PyList_GetItem(incoming, i);
				data.push_back( PyFloat_AsDouble(value) );
			}
		} else {
			throw logic_error("Passed PyObject pointer was not a list or tuple!");
		}
	}
	return data;
}

// PyObject -> Vector
vector<int> listTupleToVector_Int(PyObject* incoming) {
	vector<int> data;
	if (PyTuple_Check(incoming)) {
		for(Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
			PyObject *value = PyTuple_GetItem(incoming, i);
			data.push_back( PyFloat_AsDouble(value) );
		}
	} else {
		if (PyList_Check(incoming)) {
			for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
				PyObject *value = PyList_GetItem(incoming, i);
				data.push_back( PyFloat_AsDouble(value) );
			}
		} else {
			throw logic_error("Passed PyObject pointer was not a list or tuple!");
		}
	}
	return data;
}
