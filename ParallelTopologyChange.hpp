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

  static void PackNewAndModifiedObjects( NeighborCommunicator * const neighbor,
                                         MeshLevel * const meshLevel,
                                         array1d<localIndex> const & newNodePackList,
                                         array1d<localIndex> const & newEdgePackList,
                                         array1d<localIndex> const & newFacePackList,
                                         array1d<localIndex> const & modNodePackList,
                                         array1d<localIndex> const & modEdgePackList,
                                         array1d<localIndex> const & modFacePackList,
                                         array1d< array1d< ReferenceWrapper< localIndex_array > > > const & modifiedElements,
                                         int const commID );

  static void UnpackNewAndModifiedObjects( NeighborCommunicator * const neighbor,
                                           MeshLevel * const meshLevel,
                                           int const commID );

  static void PackNewModToGhosts( NeighborCommunicator * const neighbor,
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
                                    MeshLevel * const mesh );

};

} /* namespace lvarray */

#endif /* SRC_EXTERNALCOMPONENTS_HYDROFRACTURE_PARALLELTOPOLOGYCHANGE_HPP_ */
