add_executable(hw6_1 1.cpp)
add_executable(hw6_2 2.cpp)

target_link_libraries(hw6_1 PRIVATE matrix_utils Eigen_solver)
target_link_libraries(hw6_2 PRIVATE matrix_utils Eigen_solver)
