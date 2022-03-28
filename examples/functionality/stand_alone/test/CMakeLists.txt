set( SRC  test_schemes.cpp)

add_executable(test_schemes ${SRC})
target_link_libraries(test_schemes transfer_schemes)

install(TARGETS test_schemes DESTINATION bin)
