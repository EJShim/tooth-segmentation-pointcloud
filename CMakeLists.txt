cmake_minimum_required(VERSION 2.8)

PROJECT(ReadSTL)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

add_executable(arrange cxx_arrange_sequence.cxx)

if(VTK_LIBRARIES)
  target_link_libraries(arrange ${VTK_LIBRARIES})
else()
  target_link_libraries(arrange vtkHybrid vtkWidgets)
endif()