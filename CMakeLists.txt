cmake_minimum_required(VERSION 3.12)
project(MatrixMultiply_FAST C)

set(CMAKE_C_STANDARD 11)

add_executable(MatrixMultiply_FAST main.c mmul.c)