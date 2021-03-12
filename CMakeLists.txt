#
# Specify all headers
#
set( componentHeaders "ParallelTopologyChange.hpp"
                      "EmbeddedSurfacesParallelSynchronization.cpp" )

#
# Specify all sources
#

set( componentSources "ParallelTopologyChange.cpp" 
                      "EmbeddedSurfacesParallelSynchronization.cpp" )

set( dependencyList mesh )

if ( ENABLE_OPENMP )
  set( dependencyList ${dependencyList} openmp )
endif()

if ( ENABLE_CUDA )
  set( dependencyList ${dependencyList} cuda )
endif()

blt_add_library( NAME             GEOSX_PTP
                 SOURCES          ${componentSources}
                 HEADERS          ${componentHeaders}
                 DEPENDS_ON       ${dependencyList}
               )
