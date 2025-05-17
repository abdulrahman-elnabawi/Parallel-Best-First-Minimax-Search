cmake_minimum_required(VERSION 3.10)
project(ChessAI)

set(CMAKE_CXX_STANDARD 17)

# SFML
set(SFML_DIR "C:/Users/interface/Desktop/untitled/SFML-2.5.1-windows-gcc-7.3.0-mingw-64-bit/SFML-2.5.1/lib/cmake/SFML")
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED)

# Executable
add_executable(ChessAI main.cpp)

# Link libraries (كلها بنفس الـ signature)
target_link_libraries(ChessAI PUBLIC
        sfml-graphics
        sfml-window
        sfml-system
        )

if(OpenMP_CXX_FOUND)
    target_link_libraries(ChessAI PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(ChessAI PUBLIC -fopenmp)
endif()
