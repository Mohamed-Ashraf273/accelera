#include "utils/code_parallelizer_utils.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(code_parallelizer_utils, m) {
  m.doc() = "Code parallelizer utilities for loop extraction";

  py::class_<accelera::LoopInfo>(m, "LoopInfo")
      .def(py::init<>())
      .def_readwrite("type", &accelera::LoopInfo::type)
      .def_readwrite("start_line", &accelera::LoopInfo::start_line)
      .def_readwrite("end_line", &accelera::LoopInfo::end_line)
      .def_readwrite("code", &accelera::LoopInfo::code)
      .def("__repr__", [](const accelera::LoopInfo &info) {
        return "<LoopInfo type='" + info.type +
               "' lines=" + std::to_string(info.start_line) + "-" +
               std::to_string(info.end_line) + ">";
      });

  m.def(
      "extract_loops", &accelera::extract_loops, py::arg("filename"),
      py::arg("clang_args") = std::vector<std::string>(),
      "Extract loops from a C++ source file.\n\n"
      "Args:\n"
      "    filename (str): Path to the C++ source file\n"
      "    clang_args (list): Optional list of Clang compilation arguments\n\n"
      "Returns:\n"
      "    list[LoopInfo]: List of extracted loops");

  m.def("write_loops_to_json", &accelera::write_loops_to_json, py::arg("loops"),
        py::arg("output_file"),
        "Write extracted loops to a JSON file.\n\n"
        "Args:\n"
        "    loops (list[LoopInfo]): List of loop information\n"
        "    output_file (str): Path to the output JSON file\n\n"
        "Returns:\n"
        "    bool: True if successful, False otherwise");
}
