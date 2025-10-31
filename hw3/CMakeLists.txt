add_executable(hw3_1 1.cpp)

target_link_libraries(hw3_1 PRIVATE matrix_utils LU_solver Cholesky_solver QR_solver)