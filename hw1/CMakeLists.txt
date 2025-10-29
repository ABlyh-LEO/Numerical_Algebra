add_executable(hw1_1 1.cpp)
add_executable(hw1_2_1 2_1.cpp)
add_executable(hw1_2_2 2_2.cpp)

target_link_libraries(hw1_1 PRIVATE matrix_utils LU_solver)
target_link_libraries(hw1_2_1 PRIVATE matrix_utils Cholesky_solver)
target_link_libraries(hw1_2_2 PRIVATE matrix_utils Cholesky_solver)