add_executable(hw2_1 1.cpp)
add_executable(hw2_2 2.cpp)

target_link_libraries(hw2_1 PRIVATE matrix_utils LU_solver Condition_number)
target_link_libraries(hw2_2 PRIVATE matrix_utils Cholesky_solver Condition_number)