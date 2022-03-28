# - Try to find metis
# Once done this will define
#  METIS_FOUND - System has METIS
#  METIS_INCLUDE_DIRS - The METIS include directories
#  METIS_LIBRARIES - The libraries needed to use METIS
#  METIS_DEFINITIONS - Compiler switches required for using METIS

set(METIS_ROOT "${CMAKE_SOURCE_DIR}/extern/metis-5.1.0" CACHE PATH "METIS install directory")
if(METIS_ROOT)
    message(STATUS "METIS_ROOT ${METIS_ROOT}")
endif()

find_path(METIS_INCLUDE_DIR metis.h PATHS "${METIS_ROOT}/include")

find_library(METIS_LIBRARY metis PATHS "${METIS_ROOT}/lib")

set(METIS_LIBRARIES ${METIS_LIBRARY} )
set(METIS_INCLUDE_DIRS ${METIS_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set METIS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(
        METIS
        DEFAULT_MSG
        METIS_LIBRARY METIS_INCLUDE_DIR
)

mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARY )

if(METIS_FOUND AND NOT TARGET METIS::METIS)
    add_library(METIS::METIS UNKNOWN IMPORTED)
    set_target_properties(METIS::METIS PROPERTIES
            IMPORTED_LOCATION ${METIS_LIBRARIES}
            INTERFACE_INCLUDE_DIRECTORIES ${METIS_INCLUDE_DIR}
            )
endif()
