add_executable(hw5_1 1.cpp)
add_executable(hw5_2 2.cpp)
add_executable(hw5_3 3.cpp)

target_link_libraries(hw5_1 PRIVATE matrix_utils It_Method Sparse_utils)
target_link_libraries(hw5_2 PRIVATE matrix_utils It_Method Sparse_utils)
target_link_libraries(hw5_3 PRIVATE matrix_utils It_Method Sparse_utils)