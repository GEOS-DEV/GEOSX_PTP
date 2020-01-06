/**
 * @file ParallelTopologyChange.hpp
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

  static void SynchronizeTopologyChange( MeshLevel * const mesh,
                                         array1d<NeighborCommunicator> & neighbors,
                                         ModifiedObjectLists & modifiedObjects,
                                         ModifiedObjectLists & receivedObjects,
                                         int mpiCommOrder );

  static void PackNewAndModifiedObjectsToOwningRanks( NeighborCommunicator * const neighbor,
                                                      MeshLevel * const meshLevel,
                                                      ModifiedObjectLists const & modifiedObjects,
                                                      int const commID );

  static localIndex
  UnpackNewAndModifiedObjectsOnOwningRanks( NeighborCommunicator * const neighbor,
                                            MeshLevel * const mesh,
                                            int const commID,
                                            ModifiedObjectLists & receivedObjects );

  static void PackNewModifiedObjectsToGhosts( NeighborCommunicator * const neighbor,
                                  int commID,
                                  MeshLevel * const mesh,
                                  ModifiedObjectLists & receivedObjects );

  static void UnpackNewModToGhosts( NeighborCommunicator * const neighbor,
                                    int commID,
                                    MeshLevel * const mesh,
                                    ModifiedObjectLists & receivedObjects );


  static void updateConnectorsToFaceElems( std::set<localIndex> const & newFaceElements,
                                           FaceElementSubRegion const * const faceElemSubRegion,
                                           EdgeManager * const edgeManager  );

};

} /* namespace lvarray */

#endif /* SRC_EXTERNALCOMPONENTS_HYDROFRACTURE_PARALLELTOPOLOGYCHANGE_HPP_ */
