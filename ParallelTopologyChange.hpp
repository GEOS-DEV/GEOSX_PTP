/*
 * ParallelTopologyChange.hpp
 *
 *  Created on: Sep 20, 2018
 *      Author: settgast1
 */

#ifndef SRC_EXTERNALCOMPONENTS_HYDROFRACTURE_PARALLELTOPOLOGYCHANGE_HPP_
#define SRC_EXTERNALCOMPONENTS_HYDROFRACTURE_PARALLELTOPOLOGYCHANGE_HPP_

#include "physicsSolvers/surfaceGeneration/SurfaceGenerator.hpp"

namespace geosx
{
class MeshLevel;
class NeighborCommunicator;
struct ModifiedObjectLists;

class ParallelTopologyChange
{
public:
  ParallelTopologyChange();
  ~ParallelTopologyChange();

  static void SyncronizeTopologyChange( MeshLevel * const mesh,
                                        array1d<NeighborCommunicator> & neighbors,
                                        ModifiedObjectLists const & modifiedObjects );
};

} /* namespace lvarray */

#endif /* SRC_EXTERNALCOMPONENTS_HYDROFRACTURE_PARALLELTOPOLOGYCHANGE_HPP_ */
