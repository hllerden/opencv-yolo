cmake_minimum_required(VERSION 3.16)

project(yoloExercise LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
# find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)
find_package(OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA 11 REQUIRED)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# add_subdirectory(YOLOv8-CPP-Inference)
add_executable(yoloExercise
  main.cpp
  inference.h inference.cpp
)
target_link_libraries(yoloExercise      ${OpenCV_LIBS}
)




# target_link_libraries(MotionDetection Qt${QT_VERSION_MAJOR}::Core     ${OpenCV_LIBS}
# )

include(GNUInstallDirs)
install(TARGETS yoloExercise
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

