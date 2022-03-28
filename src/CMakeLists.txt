set( SRC
       LinearizedMELD.cpp
       LocatePoint.cpp
       MELD.cpp
       MELDThermal.cpp
       Octree.cpp
       RBF.cpp
       TransferScheme.cpp
   )

add_library(transfer_schemes ${SRC})
target_link_libraries(transfer_schemes
                         ${LAPACK_LIBRARIES}
                         ${MPI_CXX_LIBRARIES}
                     )


install(TARGETS transfer_schemes DESTINATION lib)
install(FILES ${HEADER_FILES} DESTINATION include)
