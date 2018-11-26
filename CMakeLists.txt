#
# Specify all headers
#
set(componentHeaders "ParallelTopologyChange.hpp" )

#
# Specify all sources
#

set(componentSources "ParallelTopologyChange.cpp" )

blt_add_library( NAME             GEOSX_PTP
                 SOURCES          ${componentSources}
                 HEADERS          ${componentHeaders}
                 DEPENDS_ON       mesh
               )
