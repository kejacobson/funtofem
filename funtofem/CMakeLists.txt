


include_directories(.)
include_directories(${Python_INCLUDE_DIRS})
include_directories(${NumPy_INCLUDE_DIR})

add_cython_target(TransferScheme CXX)
add_library(TransferScheme MODULE TransferScheme)
target_link_libraries(TransferScheme
                    transfer_schemes
                    ${LAPACK_LIBRARIES}
                    ${MPI_CXX_LIBRARIES})
python_extension_module(TransferScheme)

install(TARGETS TransferScheme LIBRARY DESTINATION funtofem)
