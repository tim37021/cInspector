  
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cassert>
#include <cstdio>


namespace py = pybind11;

static py::array_t<float> auto_correlation(py::array_t<float> arr, int min_lag, int max_lag, int window_size) {
    py::buffer_info arr_buf = arr.request(); const float *ptr = reinterpret_cast<const float *>(arr_buf.ptr);

    int arr_size = arr_buf.size;
    int stride = arr_buf.strides[0] / arr_buf.itemsize;

    // check parameters
    assert(arr_buf.ndim == 1);
    assert(arr_size >= window_size);

    min_lag = std::min(arr_size - window_size, min_lag);
    max_lag = std::min(arr_size - window_size, max_lag);


    auto result = py::array_t<float>(max_lag - min_lag + 1);
    py::buffer_info result_buf = result.request();
    float *ret = reinterpret_cast<float *>(result_buf.ptr);

    
    for(int i=min_lag; i<=max_lag; i++) {
        ret[i-min_lag] = 0;
        int idx = (arr_size-1)*stride;
        for(int w=0; w<window_size; w++) {
            ret[i-min_lag] += std::abs(ptr[idx]-ptr[idx-i*stride]);
            idx -= stride;
        }
        ret[i-min_lag] /= window_size;
    }

    return result;
}

template <class T>
static std::tuple<std::vector<int>, std::vector<int>> hcpeakvelly(py::array_t<T> arr, int base_samples=0) {
    py::buffer_info arr_buf = arr.request(); const T *ptr = reinterpret_cast<const T *>(arr_buf.ptr);

    // check parameters
    assert(arr_buf.ndim == 1);

    int arr_size = arr_buf.size;
    int stride = arr_buf.strides[0] / arr_buf.itemsize;

    std::vector<int> peaks;
    std::vector<int> valleys;

    bool state = false;
    int last_index = -16;
    T last;

    for(int i=0; i<arr_size; i++) {
        T val = ptr[i*stride];

        if(state) {
            // next peak
            if(last_index < 0 || last <= val) {
                last = val;
                last_index = i;
            }
            if(i - last_index > 16) {
                peaks.push_back(last_index+base_samples);
                state = false;
            }
        } else {
            // next valley
            if(last_index < 0 || last >= val) {
                last = val;
                last_index = i;
            }
            if(i - last_index > 16) {
                valleys.push_back(last_index+base_samples);
                state = true;
            }
        }

    }

    return std::make_tuple(peaks, valleys);
}

template <class T>
class hcPeakValley {
public:
    hcPeakValley(int baseSamples=0)
    {
        init(baseSamples);
    }

    std::tuple<std::vector<int>, std::vector<int>> operator()(py::array_t<T> arr) {
        py::buffer_info arr_buf = arr.request(); const T *ptr = reinterpret_cast<const T *>(arr_buf.ptr);
        // check parameters
        assert(arr_buf.ndim == 1);

        int arr_size = arr_buf.size;
        int stride = arr_buf.strides[0] / arr_buf.itemsize;

        std::vector<int> peaks;
        std::vector<int> valleys;
        
        
        for(int i=0; i<arr_size; i++) {
            T val = ptr[i*stride];

            if(m_state) {
                // next peak
                if(m_lastIndex < 0 || m_last <= val) {
                    m_last = val;
                    m_lastIndex = m_samples + i;
                }
                if(m_samples + i - m_lastIndex > 16) {
                    peaks.push_back(m_lastIndex);
                    m_state = false;
                }
            } else {
                // next valley
                if(m_lastIndex < 0 || m_last >= val) {
                    m_last = val;
                    m_lastIndex = m_samples + i;
                }
                if(m_samples + i - m_lastIndex > 16) {
                    valleys.push_back(m_lastIndex);
                    m_state = true;
                }
            }

        }

        m_samples += arr_size;

        return std::make_tuple(peaks, valleys);
    }

    void init(int baseSamples) {
        m_samples = baseSamples;
        m_lastIndex = -1;
        m_state = false;
    }

    int samples() const {
        return m_samples;
    }

private:
    int m_samples;
    T m_last;
    int m_lastIndex;
    bool m_state;
};
PYBIND11_MODULE(cInspector, m) {
    m.doc() = R"pbdoc(
        cInspector
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("auto_correlation", &auto_correlation, R"pbdoc(
        auto-correlation
        Apply auto correlation on given array
    )pbdoc");

    m.def("hcpeakvalley", &hcpeakvelly<float>, R"pbdoc(
        hcpeakvelly
        Apply hcpeakvalley algorithm on given array
    )pbdoc",
    py::arg("arr"),
    py::arg("base_samples")=0);

    py::class_<hcPeakValley<float>>(m, "hcPeakValley")
        .def(py::init<int>(), py::arg("base_samples")=0)
        .def("__call__", &hcPeakValley<float>::operator())
        .def("init", &hcPeakValley<float>::init)
        .def_property_readonly("samples", &hcPeakValley<float>::samples);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}