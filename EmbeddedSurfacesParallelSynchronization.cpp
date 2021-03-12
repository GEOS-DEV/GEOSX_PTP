/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * Copyright (c) 2018-2020 Lawrence Livermore National Security LLC
 * Copyright (c) 2018-2020 The Board of Trustees of the Leland Stanford Junior University
 * Copyright (c) 2018-2020 Total, S.A
 * Copyright (c) 2019-     GEOSX Contributors
 * All rights reserved
 *
 * See top level LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */

/**
 * @file EmebeddedSurfacesParallelSynchronization.cpp
 */

#include "EmbeddedSurfacesParallelSynchronization.hpp"

#include "common/GeosxMacros.hpp"
#include "common/TimingMacros.hpp"
#include "mesh/ElementRegionManager.hpp"
#include "mesh/ExtrinsicMeshData.hpp"
#include "mpiCommunications/CommunicationTools.hpp"


namespace geosx
{

using namespace dataRepository;

EmebeddedSurfacesParallelSynchronization::EmebeddedSurfacesParallelSynchronization()
{}

EmebeddedSurfacesParallelSynchronization::~EmebeddedSurfacesParallelSynchronization()
{}

void EmebeddedSurfacesParallelSynchronization::synchronizeNewSurfaces( MeshLevel & mesh,
                                                                       std::vector< NeighborCommunicator > & neighbors,
                                                                       NewObjectLists & newObjects,
                                                                       NewObjectLists & receivedObjects )
{

  NodeManager & nodeManager = mesh.getNodeManager();
  EdgeManager & edgeManager = mesh.getEdgeManager();
  ElementRegionManager & elemManager = mesh.getElemManager();

  //************************************************************************************************
  // We need to send over the new embedded surfaces and related objects for those whose parents are ghosts on neighbors.

  MPI_iCommData commData;
  commData.resize( neighbors.size());
  for( unsigned int neighborIndex=0; neighborIndex<neighbors.size(); ++neighborIndex )
  {
    NeighborCommunicator & neighbor = neighbors[neighborIndex];

    packNewObjectsToGhosts( &neighbor,
                            commData.commID,
                            mesh,
                            newObjects );

    neighbor.mpiISendReceiveBufferSizes( commData.commID,
                                         commData.mpiSizeSendBufferRequest[neighborIndex],
                                         commData.mpiSizeRecvBufferRequest[neighborIndex],
                                         MPI_COMM_GEOSX );

  }

  for( unsigned int count=0; count<neighbors.size(); ++count )
  {
    int neighborIndex;
    MpiWrapper::waitany( commData.size,
                         commData.mpiSizeRecvBufferRequest.data(),
                         &neighborIndex,
                         commData.mpiSizeRecvBufferStatus.data() );

    NeighborCommunicator & neighbor = neighbors[neighborIndex];

    neighbor.mpiISendReceiveBuffers( commData.commID,
                                     commData.mpiSendBufferRequest[neighborIndex],
                                     commData.mpiRecvBufferRequest[neighborIndex],
                                     MPI_COMM_GEOSX );
  }


  for( unsigned int count=0; count<neighbors.size(); ++count )
  {

    int neighborIndex = count;
    if( mpiCommOrder == 0 )
    {
      MpiWrapper::waitany( commData.size,
                           commData.mpiRecvBufferRequest.data(),
                           &neighborIndex,
                           commData.mpiRecvBufferStatus.data() );
    }
    else
    {
      MpiWrapper::wait( commData.mpiRecvBufferRequest.data() + count,
                        commData.mpiRecvBufferStatus.data() + count );
    }

    NeighborCommunicator & neighbor = neighbors[neighborIndex];

    unpackNewToGhosts( &neighbor, commData.commID, mesh, receivedObjects );
  }
}

void EmebeddedSurfacesParallelSynchronization::packNewObjectsToGhosts( NeighborCommunicator * const neighbor,
                                                                       int commID,
                                                                       MeshLevel & mesh,
                                                                       NewObjectLists & newObjects )
{
  ElementRegionManager & elemManager = mesh.getElemManager();

  ElementRegionManager::ElementViewAccessor< arrayView1d< localIndex > > newElemsToSend;
  array1d< array1d< localIndex_array > > newElemsToSendData;

  newElemsToSendData.resize( elemManager.numRegions() );
  newElemsToSend.resize( elemManager.numRegions() );
  modElemsToSendData.resize( elemManager.numRegions() );
  modElemsToSend.resize( elemManager.numRegions() );
  for( localIndex er=0; er<elemManager.numRegions(); ++er )
  {
    ElementRegionBase & elemRegion = elemManager.getRegion( er );
    newElemsToSendData[er].resize( elemRegion.numSubRegions() );
    newElemsToSend[er].resize( elemRegion.numSubRegions() );

    elemRegion.forElementSubRegionsIndex< EmbeddedSurfaceSubRegion >( [&]( localIndex const esr,
                                                                       EmbeddedSurfaceSubRegion & subRegion )
    {
      localIndex_array & elemGhostsToSend = subRegion.getNeighborData( neighbor->neighborRank() ).ghostsToSend();
      arrayView1d< integer > const & subRegionGhostRank = subRegion.ghostRank();
      for( localIndex const & k : newObjects.newElements.at( {er, esr} ) )
      {
        if( subRegionGhosRank[k] == neighbor->neighborRank() )
        {
          newElemsToSendData[er][esr].emplace_back( k );
          elemGhostsToSend.emplace_back( k );
        }
      }
      newElemsToSend[er][esr] = newElemsToSendData[er][esr];
    } );
  }

  // TODO for now I am not packing maps but eventually I ll probably need to
  int bufferSize = 0;

  bufferSize += elemManager.PackGlobalMapsSize( newElemsToSend );

  //bufferSize += elemManager.PackUpDownMapsSize( newElemsToSend );

  bufferSize += elemManager.PackSize( {}, newElemsToSend );

  //bufferSize += elemManager.PackUpDownMapsSize( modElemsToSend );

  neighbor->resizeSendBuffer( commID, bufferSize );

  buffer_type & sendBuffer = neighbor->sendBuffer( commID );
  buffer_unit_type * sendBufferPtr = sendBuffer.data();

  int packedSize = 0;

  packedSize += elemManager.PackGlobalMaps( sendBufferPtr, newElemsToSend );

  // packedSize += elemManager.PackUpDownMaps( sendBufferPtr, newElemsToSend );

  packedSize += elemManager.Pack( sendBufferPtr, {}, newElemsToSend );

  // packedSize += elemManager.PackUpDownMaps( sendBufferPtr, modElemsToSend );

  GEOSX_ERROR_IF( bufferSize != packedSize, "Allocated Buffer Size is not equal to packed buffer size" );

  neighbor->mpiISendReceive( commID, MPI_COMM_GEOSX );
}

void EmebeddedSurfacesParallelSynchronization::unpackNewToGhosts( NeighborCommunicator * const neighbor,
                                                                  int commID,
                                                                  MeshLevel & mesh,
                                                                  NewObjectLists & receivedObjects )
{
  int unpackedSize = 0;

  ElementRegionManager & elemManager = mesh.getElemManager();

  buffer_type const & receiveBuffer = neighbor->receiveBuffer( commID );
  buffer_unit_type const * receiveBufferPtr = receiveBuffer.data();

  ElementRegionManager::ElementReferenceAccessor< localIndex_array > newGhostElems;
  array1d< array1d< localIndex_array > > newGhostElemsData;
  newGhostElems.resize( elemManager.numRegions() );
  newGhostElemsData.resize( elemManager.numRegions() );
  for( localIndex er=0; er<elemManager.numRegions(); ++er )
  {
    ElementRegionBase & elemRegion = elemManager.getRegion( er );
    newGhostElemsData[er].resize( elemRegion.numSubRegions() );
    newGhostElems[er].resize( elemRegion.numSubRegions() );
    for( localIndex esr=0; esr<elemRegion.numSubRegions(); ++esr )
    {
      newGhostElems[er][esr].set( newGhostElemsData[er][esr] );
    }
  }

  // unpackedSize += elemManager.UnpackGlobalMaps( receiveBufferPtr, newGhostElems );

  // unpackedSize += elemManager.UnpackUpDownMaps( receiveBufferPtr, newGhostElems, true );

  unpackedSize += elemManager.Unpack( receiveBufferPtr, newGhostElems );

  // unpackedSize += elemManager.UnpackUpDownMaps( receiveBufferPtr, modGhostElems, true );

  elemManager.forElementSubRegionsComplete< ElementSubRegionBase >(
    [&]( localIndex const er, localIndex const esr, ElementRegionBase &, ElementSubRegionBase & subRegion )
  {
    localIndex_array & elemGhostsToReceive = subRegion.getNeighborData( neighbor->neighborRank() ).ghostsToReceive();

    if( newGhostElemsData[er][esr].size() > 0 )
    {
      elemGhostsToReceive.move( LvArray::MemorySpace::CPU );

      for( localIndex const & newElemIndex : newGhostElemsData[er][esr] )
      {
        elemGhostsToReceive.emplace_back( newElemIndex );
        receivedObjects.newElements[ { er, esr } ].insert( newElemIndex );
      }
    }
  } );
}



} /* namespace geosx */
