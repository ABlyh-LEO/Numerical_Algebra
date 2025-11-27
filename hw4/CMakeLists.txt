add_executable(hw4_1 1.cpp)
add_executable(hw4_2 2.cpp)

target_link_libraries(hw4_1 PRIVATE matrix_utils It_Method)
target_link_libraries(hw4_2 PRIVATE matrix_utils It_Method Sparse_utils)