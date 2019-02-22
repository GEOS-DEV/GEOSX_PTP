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

//  static void PackNewObjectsToOwningRank( NeighborCommunicator * const neighbor,
//                                          MeshLevel const * const meshLevel,
//                                          ModifiedObjectLists const & modifiedObjects,
//                                          int const commID );
//
//  static void PackNewObjectsToGhostRanks( NeighborCommunicator * const neighbor,
//                                          MeshLevel const * const meshLevel,
//                                          ModifiedObjectLists const & modifiedObjects,
//                                          int const commID );
//
//  static void PackModifiedObjectsToOwningRank( NeighborCommunicator * const neighbor,
//                                               MeshLevel const * const meshLevel,
//                                               ModifiedObjectLists const & modifiedObjects,
//                                               int const commID );
//
//  static void PackModifiedObjectsToGhostRanks( NeighborCommunicator * const neighbor,
//                                               MeshLevel const * const meshLevel,
//                                               ModifiedObjectLists const & modifiedObjects,
//                                               int const commID );




  static void PackNewAndModifiedObjectsToOwningRanks( NeighborCommunicator * const neighbor,
                                                      MeshLevel * const meshLevel,
                                                      ModifiedObjectLists const & modifiedObjects,
                                                      int const commID );

  static void UnpackNewAndModifiedObjects( NeighborCommunicator * const neighbor,
                                           MeshLevel * const meshLevel,
                                           int const commID );

  static void PackNewModifiedObjectsToGhosts( NeighborCommunicator * const neighbor,
                                  int commID,
                                  MeshLevel * const mesh,
                                  set<localIndex> const & allNewNodes,
                                  set<localIndex> const & allNewEdges,
                                  set<localIndex> const & allNewFaces,
                                  set<localIndex> const & allModNodes,
                                  set<localIndex> const & allModEdges,
                                  set<localIndex> const & allModFaces,
                                  array1d<array1d< set<localIndex> > > const & allModifiedElements );

  static void UnpackNewModToGhosts( NeighborCommunicator * const neighbor,
                                    int commID,
                                    MeshLevel * const mesh,
                                    ModifiedObjectLists & receivedObjects );

};

} /* namespace lvarray */

#endif /* SRC_EXTERNALCOMPONENTS_HYDROFRACTURE_PARALLELTOPOLOGYCHANGE_HPP_ */
