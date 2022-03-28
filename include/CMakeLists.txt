set( HEADER_FILES
        LinearizedMELD.h
        LocatePoint.h
        MELD.h
        MELDThermal.h
        Octree.h
        RBF.h
        TransferScheme.h
        funtofemlapack.h
   )

install(FILES ${HEADER_FILES} DESTINATION include)
