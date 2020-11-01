cmake_minimum_required(VERSION 3.4)
project(inspector)

if(MSVC)
add_compile_options(/wd4244)
endif()

add_subdirectory(pybind11)
pybind11_add_module(cInspector src/main.cpp)

add_custom_command(TARGET cInspector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cInspector> ${CMAKE_SOURCE_DIR}/test
    COMMAND ${PYTHON_EXECUTABLE} -m pytest -v ${CMAKE_SOURCE_DIR}/test/test.py
)