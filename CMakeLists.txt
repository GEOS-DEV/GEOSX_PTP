#
# Specify all headers
#
set( componentHeaders "ParallelTopologyChange.hpp" )

#
# Specify all sources
#

set( componentSources "ParallelTopologyChange.cpp" )

set( dependencyList mesh )

if ( ENABLE_OPENMP )
  set( dependencyList ${dependencyList} openmp )
endif()

if ( ENABLE_CUDA )
  set( dependencyList ${dependencyList} cuda )
endif()

if ( ENABLE_HYPRE )
  set( dependencyList ${dependencyList} hypre )
endif()

blt_add_library( NAME             GEOSX_PTP
                 SOURCES          ${componentSources}
                 HEADERS          ${componentHeaders}
                 DEPENDS_ON       ${dependencyList}
               )
