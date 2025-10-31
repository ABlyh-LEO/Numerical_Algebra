add_executable(hw3_1 1.cpp)
add_executable(hw3_2 2.cpp)
add_executable(hw3_3 3.cpp)

target_link_libraries(hw3_1 PRIVATE matrix_utils LU_solver Cholesky_solver QR_solver)
target_link_libraries(hw3_2 PRIVATE matrix_utils QR_solver)
target_link_libraries(hw3_3 PRIVATE matrix_utils QR_solver)