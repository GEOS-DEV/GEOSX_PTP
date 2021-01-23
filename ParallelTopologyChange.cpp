/**
 * @file ParallelTopologyChange.cpp
 */

#include "ParallelTopologyChange.hpp"

#include "common/GeosxMacros.hpp"
#include "common/TimingMacros.hpp"
#include "mesh/ElementRegionManager.hpp"
#include "mesh/ExtrinsicMeshData.hpp"
#include "mpiCommunications/CommunicationTools.hpp"


namespace geosx
{

using namespace dataRepository;

ParallelTopologyChange::ParallelTopologyChange()
{}

ParallelTopologyChange::~ParallelTopologyChange()
{}

void ParallelTopologyChange::synchronizeTopologyChange( MeshLevel * const mesh,
                                                        std::vector< NeighborCommunicator > & neighbors,
                                                        ModifiedObjectLists & modifiedObjects,
                                                        ModifiedObjectLists & receivedObjects,
                                                        int mpiCommOrder )
{

  NodeManager * const nodeManager = mesh->getNodeManager();
  EdgeManager * const edgeManager = mesh->getEdgeManager();
  FaceManager * const faceManager = mesh->getFaceManager();
  ElementRegionManager * const elemManager = mesh->getElemManager();

  //************************************************************************************************
  // 2) first we need to send over:
  //   a) the new objects to owning ranks. New objects are assumed to be
  //      owned by the rank that owned the parent.
  //   b) the modified objects to the owning ranks.

  // pack the buffers, and send the size of the buffers
  MPI_iCommData commData;
  commData.resize( neighbors.size() );
  for( unsigned int neighborIndex=0; neighborIndex<neighbors.size(); ++neighborIndex )
  {
    NeighborCommunicator & neighbor = neighbors[neighborIndex];

    packNewAndModifiedObjectsToOwningRanks( &neighbor,
                                            mesh,
                                            modifiedObjects,
                                            commData.commID );

    neighbor.mpiISendReceiveBufferSizes( commData.commID,
                                         commData.mpiSizeSendBufferRequest[neighborIndex],
                                         commData.mpiSizeRecvBufferRequest[neighborIndex],
                                         MPI_COMM_GEOSX );

  }

  // send/recv the buffers
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

  // unpack the buffers and get lists of the new objects.
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
    // Unpack buffers in set ordering for integration testing
    else
    {
      MpiWrapper::wait( commData.mpiRecvBufferRequest.data() + count,
                        commData.mpiRecvBufferStatus.data() + count );
    }

    NeighborCommunicator & neighbor = neighbors[neighborIndex];

    unpackNewAndModifiedObjectsOnOwningRanks( &neighbor,
                                              mesh,
                                              commData.commID,
                                              receivedObjects );
  }

  nodeManager->inheritGhostRankFromParent( receivedObjects.newNodes );
  edgeManager->inheritGhostRankFromParent( receivedObjects.newEdges );
  faceManager->inheritGhostRankFromParent( receivedObjects.newFaces );

  elemManager->forElementSubRegionsComplete< FaceElementSubRegion >( [&]( localIndex const er,
                                                                          localIndex const esr,
                                                                          ElementRegionBase &,
                                                                          FaceElementSubRegion & subRegion )
  {
    subRegion.inheritGhostRankFromParentFace( faceManager, receivedObjects.newElements[{er, esr}] );
  } );

  MpiWrapper::waitall( commData.size,
                       commData.mpiSizeSendBufferRequest.data(),
                       commData.mpiSizeSendBufferStatus.data() );

  MpiWrapper::waitall( commData.size,
                       commData.mpiSendBufferRequest.data(),
                       commData.mpiSizeSendBufferStatus.data() );

  modifiedObjects.insert( receivedObjects );


  //************************************************************************************************
  // 3) now we need to send over:
  //   a) the new objects whose parents are ghosts on neighbors.
  //   b) the modified objects whose parents are ghosts on neighbors.



  MPI_iCommData commData2;
  commData2.resize( neighbors.size());
  for( unsigned int neighborIndex=0; neighborIndex<neighbors.size(); ++neighborIndex )
  {
    NeighborCommunicator & neighbor = neighbors[neighborIndex];

    packNewModifiedObjectsToGhosts( &neighbor,
                                    commData2.commID,
                                    mesh,
                                    modifiedObjects );

    neighbor.mpiISendReceiveBufferSizes( commData2.commID,
                                         commData2.mpiSizeSendBufferRequest[neighborIndex],
                                         commData2.mpiSizeRecvBufferRequest[neighborIndex],
                                         MPI_COMM_GEOSX );

  }

  for( unsigned int count=0; count<neighbors.size(); ++count )
  {
    int neighborIndex;
    MpiWrapper::waitany( commData2.size,
                         commData2.mpiSizeRecvBufferRequest.data(),
                         &neighborIndex,
                         commData2.mpiSizeRecvBufferStatus.data() );

    NeighborCommunicator & neighbor = neighbors[neighborIndex];

    neighbor.mpiISendReceiveBuffers( commData2.commID,
                                     commData2.mpiSendBufferRequest[neighborIndex],
                                     commData2.mpiRecvBufferRequest[neighborIndex],
                                     MPI_COMM_GEOSX );
  }


  for( unsigned int count=0; count<neighbors.size(); ++count )
  {

    int neighborIndex = count;
    if( mpiCommOrder == 0 )
    {
      MpiWrapper::waitany( commData2.size,
                           commData2.mpiRecvBufferRequest.data(),
                           &neighborIndex,
                           commData2.mpiRecvBufferStatus.data() );
    }
    else
    {
      MpiWrapper::wait( commData2.mpiRecvBufferRequest.data() + count,
                        commData2.mpiRecvBufferStatus.data() + count );
    }

    NeighborCommunicator & neighbor = neighbors[neighborIndex];


    unpackNewModToGhosts( &neighbor, commData2.commID, mesh, receivedObjects );
  }

  modifiedObjects.insert( receivedObjects );

  nodeManager->fixUpDownMaps( false );
  edgeManager->fixUpDownMaps( false );
  faceManager->fixUpDownMaps( false );


  for( localIndex er=0; er<elemManager->numRegions(); ++er )
  {
    ElementRegionBase * const elemRegion = elemManager->getRegion( er );
    for( localIndex esr=0; esr<elemRegion->numSubRegions(); ++esr )
    {
      ElementSubRegionBase * const subRegion = elemRegion->getSubRegion( esr );
      subRegion->fixUpDownMaps( false );
    }
  }

  elemManager->forElementSubRegionsComplete< FaceElementSubRegion >( [&]( localIndex const er,
                                                                          localIndex const esr,
                                                                          ElementRegionBase const &,
                                                                          FaceElementSubRegion const & subRegion )
  {
    updateConnectorsToFaceElems( receivedObjects.newElements.at( {er, esr} ),
                                 &subRegion,
                                 edgeManager );
  } );


  std::set< localIndex > allTouchedNodes;
  allTouchedNodes.insert( modifiedObjects.newNodes.begin(), modifiedObjects.newNodes.end() );
  allTouchedNodes.insert( modifiedObjects.modifiedNodes.begin(), modifiedObjects.modifiedNodes.end() );
  nodeManager->depopulateUpMaps( allTouchedNodes,
                                 edgeManager->nodeList(),
                                 faceManager->nodeList().toViewConst(),
                                 *elemManager );

  std::set< localIndex > allTouchedEdges;
  allTouchedEdges.insert( modifiedObjects.newEdges.begin(), modifiedObjects.newEdges.end() );
  allTouchedEdges.insert( modifiedObjects.modifiedEdges.begin(), modifiedObjects.modifiedEdges.end() );
  edgeManager->depopulateUpMaps( allTouchedEdges,
                                 faceManager->edgeList().toViewConst() );

  std::set< localIndex > allTouchedFaces;
  allTouchedFaces.insert( modifiedObjects.newFaces.begin(), modifiedObjects.newFaces.end() );
  allTouchedFaces.insert( modifiedObjects.modifiedFaces.begin(), modifiedObjects.modifiedFaces.end() );
  faceManager->depopulateUpMaps( allTouchedFaces,
                                 *elemManager );

  nodeManager->enforceStateFieldConsistencyPostTopologyChange( modifiedObjects.modifiedNodes );
  edgeManager->enforceStateFieldConsistencyPostTopologyChange( modifiedObjects.modifiedEdges );
  faceManager->enforceStateFieldConsistencyPostTopologyChange( modifiedObjects.modifiedFaces );


}


void
ParallelTopologyChange::
  packNewAndModifiedObjectsToOwningRanks( NeighborCommunicator * const neighbor,
                                          MeshLevel * const meshLevel,
                                          ModifiedObjectLists const & modifiedObjects,
                                          int const commID )
{
  int bufferSize = 0;

  NodeManager & nodeManager = *(meshLevel->getNodeManager());
  EdgeManager & edgeManager = *(meshLevel->getEdgeManager());
  FaceManager & faceManager = *(meshLevel->getFaceManager());
  ElementRegionManager & elemManager = *(meshLevel->getElemManager() );

  arrayView1d< integer > const & nodeGhostRank = nodeManager.ghostRank();
  arrayView1d< integer > const & edgeGhostRank = edgeManager.ghostRank();
  arrayView1d< integer > const & faceGhostRank = faceManager.ghostRank();

  arrayView1d< localIndex > const & parentNodeIndices = nodeManager.getExtrinsicData< extrinsicMeshData::ParentIndex >();
  arrayView1d< localIndex > const & parentEdgeIndices = edgeManager.getExtrinsicData< extrinsicMeshData::ParentIndex >();
  arrayView1d< localIndex > const & parentFaceIndices = faceManager.getExtrinsicData< extrinsicMeshData::ParentIndex >();

  int const neighborRank = neighbor->neighborRank();

  array1d< localIndex > newNodePackListArray( modifiedObjects.newNodes.size());
  {
    localIndex a=0;
    for( auto const index : modifiedObjects.newNodes )
    {
      localIndex const parentNodeIndex = ObjectManagerBase::getParentRecusive( parentNodeIndices, index );
      if( nodeGhostRank[parentNodeIndex] == neighborRank )
      {
        newNodePackListArray[a] = index;
        ++a;
      }
    }
    newNodePackListArray.resize( a );
  }
  array1d< localIndex > modNodePackListArray( modifiedObjects.modifiedNodes.size());
  {
    localIndex a=0;
    for( auto const index : modifiedObjects.modifiedNodes )
    {
      localIndex const parentNodeIndex = ObjectManagerBase::getParentRecusive( parentNodeIndices, index );
      if( nodeGhostRank[parentNodeIndex] == neighborRank )
      {
        modNodePackListArray[a] = index;
        ++a;
      }
    }
    modNodePackListArray.resize( a );
  }


  array1d< localIndex > newEdgePackListArray( modifiedObjects.newEdges.size());
  {
    localIndex a=0;
    for( auto const index : modifiedObjects.newEdges )
    {
      localIndex const parentIndex = ObjectManagerBase::getParentRecusive( parentEdgeIndices, index );
      if( edgeGhostRank[parentIndex] == neighborRank )
      {
        newEdgePackListArray[a] = index;
        ++a;
      }
    }
    newEdgePackListArray.resize( a );
  }
  array1d< localIndex > modEdgePackListArray( modifiedObjects.modifiedEdges.size());
  {
    localIndex a=0;
    for( auto const index : modifiedObjects.modifiedEdges )
    {
      localIndex const parentIndex = ObjectManagerBase::getParentRecusive( parentEdgeIndices, index );
      if( edgeGhostRank[parentIndex] == neighborRank )
      {
        modEdgePackListArray[a] = index;
        ++a;
      }
    }
    modEdgePackListArray.resize( a );
  }


  array1d< localIndex > newFacePackListArray( modifiedObjects.newFaces.size());
  {
    localIndex a=0;
    for( auto const index : modifiedObjects.newFaces )
    {
      localIndex const parentIndex = ObjectManagerBase::getParentRecusive( parentFaceIndices, index );
      if( faceGhostRank[parentIndex] == neighborRank )
      {
        newFacePackListArray[a] = index;
        ++a;
      }
    }
    newFacePackListArray.resize( a );
  }
  array1d< localIndex > modFacePackListArray( modifiedObjects.modifiedFaces.size());
  {
    localIndex a=0;
    for( auto const index : modifiedObjects.modifiedFaces )
    {
      localIndex const parentIndex = ObjectManagerBase::getParentRecusive( parentFaceIndices, index );
      if( faceGhostRank[parentIndex] == neighborRank )
      {
        modFacePackListArray[a] = index;
        ++a;
      }
    }
    modFacePackListArray.resize( a );
  }


  ElementRegionManager::ElementViewAccessor< arrayView1d< localIndex > > newElemPackList;
  array1d< array1d< localIndex_array > > newElemData;
  ElementRegionManager::ElementReferenceAccessor< localIndex_array > modElemPackList;
  array1d< array1d< localIndex_array > > modElemData;
  newElemPackList.resize( elemManager.numRegions());
  newElemData.resize( elemManager.numRegions());
  modElemPackList.resize( elemManager.numRegions());
  modElemData.resize( elemManager.numRegions());
  for( localIndex er=0; er<elemManager.numRegions(); ++er )
  {
    ElementRegionBase * const elemRegion = elemManager.getRegion( er );
    newElemPackList[er].resize( elemRegion->numSubRegions());
    newElemData[er].resize( elemRegion->numSubRegions());
    modElemPackList[er].resize( elemRegion->numSubRegions());
    modElemData[er].resize( elemRegion->numSubRegions());
    for( localIndex esr=0; esr<elemRegion->numSubRegions(); ++esr )
    {
      ElementSubRegionBase * const subRegion = elemRegion->getSubRegion( esr );
      arrayView1d< integer > const & subRegionGhostRank = subRegion->ghostRank();
      if( modifiedObjects.modifiedElements.count( {er, esr} ) > 0 )
      {
        std::set< localIndex > const & elemList = modifiedObjects.modifiedElements.at( {er, esr} );

        modElemData[er][esr].resize( elemList.size() );

        localIndex a=0;
        for( auto const index : elemList )
        {
          if( subRegionGhostRank[index] == neighborRank )
          {
            modElemData[er][esr][a] = index;
            ++a;
          }
        }
        modElemData[er][esr].resize( a );
      }

      if( modifiedObjects.newElements.count( {er, esr} ) > 0 )
      {
        std::set< localIndex > const & elemList = modifiedObjects.newElements.at( {er, esr} );

        newElemData[er][esr].resize( elemList.size() );

        localIndex a=0;
        for( auto const index : elemList )
        {
          if( subRegionGhostRank[index] == neighborRank )
          {
            newElemData[er][esr][a] = index;
            ++a;
          }
        }
        newElemData[er][esr].resize( a );
      }

      newElemPackList[er][esr] = newElemData[er][esr];
      modElemPackList[er][esr].set( modElemData[er][esr] );
    }
  }



  bufferSize += nodeManager.packGlobalMapsSize( newNodePackListArray, 0 );
  bufferSize += edgeManager.packGlobalMapsSize( newEdgePackListArray, 0 );
  bufferSize += faceManager.packGlobalMapsSize( newFacePackListArray, 0 );
  bufferSize += elemManager.PackGlobalMapsSize( newElemPackList );

  bufferSize += nodeManager.packUpDownMapsSize( newNodePackListArray );
  bufferSize += edgeManager.packUpDownMapsSize( newEdgePackListArray );
  bufferSize += faceManager.packUpDownMapsSize( newFacePackListArray );
  bufferSize += elemManager.PackUpDownMapsSize( newElemPackList );

  bufferSize += nodeManager.packParentChildMapsSize( newNodePackListArray );
  bufferSize += edgeManager.packParentChildMapsSize( newEdgePackListArray );
  bufferSize += faceManager.packParentChildMapsSize( newFacePackListArray );

  bufferSize += nodeManager.packSize( {}, newNodePackListArray, 0 );
  bufferSize += edgeManager.packSize( {}, newEdgePackListArray, 0 );
  bufferSize += faceManager.packSize( {}, newFacePackListArray, 0 );
  bufferSize += elemManager.PackSize( {}, newElemPackList );

  bufferSize += nodeManager.packUpDownMapsSize( modNodePackListArray );
  bufferSize += edgeManager.packUpDownMapsSize( modEdgePackListArray );
  bufferSize += faceManager.packUpDownMapsSize( modFacePackListArray );
  bufferSize += elemManager.PackUpDownMapsSize( modElemPackList );

  bufferSize += nodeManager.packParentChildMapsSize( modNodePackListArray );
  bufferSize += edgeManager.packParentChildMapsSize( modEdgePackListArray );
  bufferSize += faceManager.packParentChildMapsSize( modFacePackListArray );

  bufferSize += nodeManager.packSize( {}, modNodePackListArray, 0 );
  bufferSize += edgeManager.packSize( {}, modEdgePackListArray, 0 );
  bufferSize += faceManager.packSize( {}, modFacePackListArray, 0 );


  neighbor->resizeSendBuffer( commID, bufferSize );

  buffer_type & sendBuffer = neighbor->sendBuffer( commID );
  buffer_unit_type * sendBufferPtr = sendBuffer.data();

  int packedSize = 0;

  packedSize += nodeManager.packGlobalMaps( sendBufferPtr, newNodePackListArray, 0 );
  packedSize += edgeManager.packGlobalMaps( sendBufferPtr, newEdgePackListArray, 0 );
  packedSize += faceManager.packGlobalMaps( sendBufferPtr, newFacePackListArray, 0 );
  packedSize += elemManager.PackGlobalMaps( sendBufferPtr, newElemPackList );

  packedSize += nodeManager.packUpDownMaps( sendBufferPtr, newNodePackListArray );
  packedSize += edgeManager.packUpDownMaps( sendBufferPtr, newEdgePackListArray );
  packedSize += faceManager.packUpDownMaps( sendBufferPtr, newFacePackListArray );
  packedSize += elemManager.PackUpDownMaps( sendBufferPtr, newElemPackList );

  packedSize += nodeManager.packParentChildMaps( sendBufferPtr, newNodePackListArray );
  packedSize += edgeManager.packParentChildMaps( sendBufferPtr, newEdgePackListArray );
  packedSize += faceManager.packParentChildMaps( sendBufferPtr, newFacePackListArray );

  packedSize += nodeManager.pack( sendBufferPtr, {}, newNodePackListArray, 0 );
  packedSize += edgeManager.pack( sendBufferPtr, {}, newEdgePackListArray, 0 );
  packedSize += faceManager.pack( sendBufferPtr, {}, newFacePackListArray, 0 );
  packedSize += elemManager.Pack( sendBufferPtr, {}, newElemPackList );

  packedSize += nodeManager.packUpDownMaps( sendBufferPtr, modNodePackListArray );
  packedSize += edgeManager.packUpDownMaps( sendBufferPtr, modEdgePackListArray );
  packedSize += faceManager.packUpDownMaps( sendBufferPtr, modFacePackListArray );
  packedSize += elemManager.PackUpDownMaps( sendBufferPtr, modElemPackList );

  packedSize += nodeManager.packParentChildMaps( sendBufferPtr, modNodePackListArray );
  packedSize += edgeManager.packParentChildMaps( sendBufferPtr, modEdgePackListArray );
  packedSize += faceManager.packParentChildMaps( sendBufferPtr, modFacePackListArray );

  packedSize += nodeManager.pack( sendBufferPtr, {}, modNodePackListArray, 0 );
  packedSize += edgeManager.pack( sendBufferPtr, {}, modEdgePackListArray, 0 );
  packedSize += faceManager.pack( sendBufferPtr, {}, modFacePackListArray, 0 );


  GEOSX_ERROR_IF( bufferSize != packedSize,
                  "Allocated Buffer Size ("<<bufferSize<<") is not equal to packed buffer size("<<packedSize<<")" );


}

localIndex
ParallelTopologyChange::
  unpackNewAndModifiedObjectsOnOwningRanks( NeighborCommunicator * const neighbor,
                                            MeshLevel * const mesh,
                                            int const commID,
//                                          array1d<array1d< std::set<localIndex> > > & allNewElements,
//                                          array1d<array1d< std::set<localIndex> > > & allModifiedElements,
                                            ModifiedObjectLists & receivedObjects )
{
  GEOSX_MARK_FUNCTION;

  NodeManager * const nodeManager = mesh->getNodeManager();
  EdgeManager * const edgeManager = mesh->getEdgeManager();
  FaceManager * const faceManager = mesh->getFaceManager();
  ElementRegionManager * const elemManager = mesh->getElemManager();



  buffer_type const & receiveBuffer = neighbor->receiveBuffer( commID );
  buffer_unit_type const * receiveBufferPtr = receiveBuffer.data();
  localIndex_array newLocalNodes, modifiedLocalNodes;
  localIndex_array newLocalEdges, modifiedLocalEdges;
  localIndex_array newLocalFaces, modifiedLocalFaces;

  ElementRegionManager::ElementReferenceAccessor< array1d< localIndex > > newLocalElements;
  array1d< array1d< localIndex_array > > newLocalElementsData;

  ElementRegionManager::ElementReferenceAccessor< localIndex_array > modifiedLocalElements;
  array1d< array1d< localIndex_array > > modifiedLocalElementsData;

  newLocalElements.resize( elemManager->numRegions());
  newLocalElementsData.resize( elemManager->numRegions());
  modifiedLocalElements.resize( elemManager->numRegions());
  modifiedLocalElementsData.resize( elemManager->numRegions());
  for( localIndex er=0; er<elemManager->numRegions(); ++er )
  {
    ElementRegionBase * const elemRegion = elemManager->getRegion( er );
    newLocalElements[er].resize( elemRegion->numSubRegions());
    newLocalElementsData[er].resize( elemRegion->numSubRegions());
    modifiedLocalElements[er].resize( elemRegion->numSubRegions());
    modifiedLocalElementsData[er].resize( elemRegion->numSubRegions());
    for( localIndex esr=0; esr<elemRegion->numSubRegions(); ++esr )
    {
      newLocalElements[er][esr].set( newLocalElementsData[er][esr] );
      modifiedLocalElements[er][esr].set( modifiedLocalElementsData[er][esr] );
    }
  }



  int unpackedSize = 0;
  unpackedSize += nodeManager->unpackGlobalMaps( receiveBufferPtr, newLocalNodes, 0 );
  unpackedSize += edgeManager->unpackGlobalMaps( receiveBufferPtr, newLocalEdges, 0 );
  unpackedSize += faceManager->unpackGlobalMaps( receiveBufferPtr, newLocalFaces, 0 );
  unpackedSize += elemManager->UnpackGlobalMaps( receiveBufferPtr, newLocalElements );

  unpackedSize += nodeManager->unpackUpDownMaps( receiveBufferPtr, newLocalNodes, true, true );
  unpackedSize += edgeManager->unpackUpDownMaps( receiveBufferPtr, newLocalEdges, true, true );
  unpackedSize += faceManager->unpackUpDownMaps( receiveBufferPtr, newLocalFaces, true, true );
  unpackedSize += elemManager->UnpackUpDownMaps( receiveBufferPtr, newLocalElements, true );

  unpackedSize += nodeManager->unpackParentChildMaps( receiveBufferPtr, newLocalNodes );
  unpackedSize += edgeManager->unpackParentChildMaps( receiveBufferPtr, newLocalEdges );
  unpackedSize += faceManager->unpackParentChildMaps( receiveBufferPtr, newLocalFaces );

  unpackedSize += nodeManager->unpack( receiveBufferPtr, newLocalNodes, 0 );
  unpackedSize += edgeManager->unpack( receiveBufferPtr, newLocalEdges, 0 );
  unpackedSize += faceManager->unpack( receiveBufferPtr, newLocalFaces, 0 );
  unpackedSize += elemManager->Unpack( receiveBufferPtr, newLocalElements );

  unpackedSize += nodeManager->unpackUpDownMaps( receiveBufferPtr, modifiedLocalNodes, false, true );
  unpackedSize += edgeManager->unpackUpDownMaps( receiveBufferPtr, modifiedLocalEdges, false, true );
  unpackedSize += faceManager->unpackUpDownMaps( receiveBufferPtr, modifiedLocalFaces, false, true );
  unpackedSize += elemManager->UnpackUpDownMaps( receiveBufferPtr, modifiedLocalElements, true );

  unpackedSize += nodeManager->unpackParentChildMaps( receiveBufferPtr, modifiedLocalNodes );
  unpackedSize += edgeManager->unpackParentChildMaps( receiveBufferPtr, modifiedLocalEdges );
  unpackedSize += faceManager->unpackParentChildMaps( receiveBufferPtr, modifiedLocalFaces );

  unpackedSize += nodeManager->unpack( receiveBufferPtr, modifiedLocalNodes, 0 );
  unpackedSize += edgeManager->unpack( receiveBufferPtr, modifiedLocalEdges, 0 );
  unpackedSize += faceManager->unpack( receiveBufferPtr, modifiedLocalFaces, 0 );
//    unpackedSize += elemManager->Unpack( receiveBufferPtr, modifiedElements );



  std::set< localIndex > & allNewNodes      = receivedObjects.newNodes;
  std::set< localIndex > & allModifiedNodes = receivedObjects.modifiedNodes;
  std::set< localIndex > & allNewEdges      = receivedObjects.newEdges;
  std::set< localIndex > & allModifiedEdges = receivedObjects.modifiedEdges;
  std::set< localIndex > & allNewFaces      = receivedObjects.newFaces;
  std::set< localIndex > & allModifiedFaces = receivedObjects.modifiedFaces;
  map< std::pair< localIndex, localIndex >, std::set< localIndex > > & allNewElements = receivedObjects.newElements;
  map< std::pair< localIndex, localIndex >, std::set< localIndex > > & allModifiedElements = receivedObjects.modifiedElements;

  allNewNodes.insert( newLocalNodes.begin(), newLocalNodes.end() );
  allModifiedNodes.insert( modifiedLocalNodes.begin(), modifiedLocalNodes.end() );

  allNewEdges.insert( newLocalEdges.begin(), newLocalEdges.end() );
  allModifiedEdges.insert( modifiedLocalEdges.begin(), modifiedLocalEdges.end() );

  allNewFaces.insert( newLocalFaces.begin(), newLocalFaces.end() );
  allModifiedFaces.insert( modifiedLocalFaces.begin(), modifiedLocalFaces.end() );

  for( localIndex er=0; er<elemManager->numRegions(); ++er )
  {
    ElementRegionBase * const elemRegion = elemManager->getRegion( er );
    for( localIndex esr=0; esr<elemRegion->numSubRegions(); ++esr )
    {
      allNewElements[{er, esr}].insert( newLocalElements[er][esr].get().begin(),
                                        newLocalElements[er][esr].get().end() );
      allModifiedElements[{er, esr}].insert( modifiedLocalElements[er][esr].get().begin(),
                                             modifiedLocalElements[er][esr].get().end() );
    }
  }

  return unpackedSize;
}


static void FilterNewObjectsForPackToGhosts( std::set< localIndex > const & objectList,
                                             arrayView1d< localIndex > const & parentIndices,
                                             localIndex_array & ghostsToSend,
                                             localIndex_array & objectsToSend )
{

  ghostsToSend.move( LvArray::MemorySpace::CPU );
  //TODO this needs to be inverted since the ghostToSend list should be much longer....
  // and the objectList is a searchable set.
  for( auto const index : objectList )
  {
    localIndex const parentIndex = parentIndices[index];
    for( localIndex a=0; a<ghostsToSend.size(); ++a )
    {
      if( ghostsToSend[a]==parentIndex )
      {
        objectsToSend.emplace_back( index );
        ghostsToSend.emplace_back( index );
        break;
      }
    }
  }
}

static void FilterModObjectsForPackToGhosts( std::set< localIndex > const & objectList,
                                             localIndex_array const & ghostsToSend,
                                             localIndex_array & objectsToSend )
{
  ghostsToSend.move( LvArray::MemorySpace::CPU );
  for( localIndex a=0; a<ghostsToSend.size(); ++a )
  {
    if( objectList.count( ghostsToSend[a] ) > 0 )
    {
      objectsToSend.emplace_back( ghostsToSend[a] );
    }
  }
}

void ParallelTopologyChange::packNewModifiedObjectsToGhosts( NeighborCommunicator * const neighbor,
                                                             int commID,
                                                             MeshLevel * const mesh,
                                                             ModifiedObjectLists & receivedObjects )
{
  NodeManager * const nodeManager = mesh->getNodeManager();
  EdgeManager * const edgeManager = mesh->getEdgeManager();
  FaceManager * const faceManager = mesh->getFaceManager();
  ElementRegionManager * const elemManager = mesh->getElemManager();

  localIndex_array newNodesToSend;
  localIndex_array newEdgesToSend;
  localIndex_array newFacesToSend;
  ElementRegionManager::ElementViewAccessor< arrayView1d< localIndex > > newElemsToSend;
  array1d< array1d< localIndex_array > > newElemsToSendData;

  localIndex_array modNodesToSend;
  localIndex_array modEdgesToSend;
  localIndex_array modFacesToSend;
  ElementRegionManager::ElementReferenceAccessor< localIndex_array > modElemsToSend;
  array1d< array1d< localIndex_array > > modElemsToSendData;

  localIndex_array & nodeGhostsToSend = nodeManager->getNeighborData( neighbor->neighborRank() ).ghostsToSend();
  localIndex_array & edgeGhostsToSend = edgeManager->getNeighborData( neighbor->neighborRank() ).ghostsToSend();
  localIndex_array & faceGhostsToSend = faceManager->getNeighborData( neighbor->neighborRank() ).ghostsToSend();

  arrayView1d< localIndex > const & nodalParentIndices = nodeManager->getExtrinsicData< extrinsicMeshData::ParentIndex >();
  arrayView1d< localIndex > const & edgeParentIndices = edgeManager->getExtrinsicData< extrinsicMeshData::ParentIndex >();
  arrayView1d< localIndex > const & faceParentIndices = faceManager->getExtrinsicData< extrinsicMeshData::ParentIndex >();

  FilterNewObjectsForPackToGhosts( receivedObjects.newNodes, nodalParentIndices, nodeGhostsToSend, newNodesToSend );
  FilterModObjectsForPackToGhosts( receivedObjects.modifiedNodes, nodeGhostsToSend, modNodesToSend );

  FilterNewObjectsForPackToGhosts( receivedObjects.newEdges, edgeParentIndices, edgeGhostsToSend, newEdgesToSend );
  FilterModObjectsForPackToGhosts( receivedObjects.modifiedEdges, edgeGhostsToSend, modEdgesToSend );

  FilterNewObjectsForPackToGhosts( receivedObjects.newFaces, faceParentIndices, faceGhostsToSend, newFacesToSend );
  FilterModObjectsForPackToGhosts( receivedObjects.modifiedFaces, faceGhostsToSend, modFacesToSend );

  SortedArray< localIndex > faceGhostsToSendSet;
  for( localIndex const & kf : faceGhostsToSend )
  {
    faceGhostsToSendSet.insert( kf );
  }

  newElemsToSendData.resize( elemManager->numRegions() );
  newElemsToSend.resize( elemManager->numRegions() );
  modElemsToSendData.resize( elemManager->numRegions() );
  modElemsToSend.resize( elemManager->numRegions() );
  for( localIndex er=0; er<elemManager->numRegions(); ++er )
  {
    ElementRegionBase * const elemRegion = elemManager->getRegion( er );
    newElemsToSendData[er].resize( elemRegion->numSubRegions() );
    newElemsToSend[er].resize( elemRegion->numSubRegions() );
    modElemsToSendData[er].resize( elemRegion->numSubRegions() );
    modElemsToSend[er].resize( elemRegion->numSubRegions() );

    elemRegion->forElementSubRegionsIndex< FaceElementSubRegion >( [&]( localIndex const esr,
                                                                        FaceElementSubRegion & subRegion )
    {
      FaceElementSubRegion::FaceMapType const & faceList = subRegion.faceList();
      localIndex_array & elemGhostsToSend = subRegion.getNeighborData( neighbor->neighborRank() ).ghostsToSend();
      for( localIndex const & k : receivedObjects.newElements.at( {er, esr} ) )
      {
        if( faceGhostsToSendSet.count( faceList( k, 0 ) ) )
        {
          newElemsToSendData[er][esr].emplace_back( k );
          elemGhostsToSend.emplace_back( k );
        }
      }
      newElemsToSend[er][esr] = newElemsToSendData[er][esr];
    } );

    elemRegion->forElementSubRegionsIndex< ElementSubRegionBase >( [&]( localIndex const esr,
                                                                        ElementSubRegionBase const & subRegion )
    {
      modElemsToSend[er][esr].set( modElemsToSendData[er][esr] );
      arrayView1d< localIndex const > const & elemGhostsToSend = subRegion.getNeighborData( neighbor->neighborRank() ).ghostsToSend();
      for( localIndex const ghostToSend : elemGhostsToSend )
      {
        if( receivedObjects.modifiedElements.at( { er, esr } ).count( ghostToSend ) > 0 )
        {
          modElemsToSendData[er][esr].emplace_back( ghostToSend );
        }
      }
    } );
  }


  int bufferSize = 0;

  bufferSize += nodeManager->packGlobalMapsSize( newNodesToSend, 0 );
  bufferSize += edgeManager->packGlobalMapsSize( newEdgesToSend, 0 );
  bufferSize += faceManager->packGlobalMapsSize( newFacesToSend, 0 );
  bufferSize += elemManager->PackGlobalMapsSize( newElemsToSend );

  bufferSize += nodeManager->packUpDownMapsSize( newNodesToSend );
  bufferSize += edgeManager->packUpDownMapsSize( newEdgesToSend );
  bufferSize += faceManager->packUpDownMapsSize( newFacesToSend );
  bufferSize += elemManager->PackUpDownMapsSize( newElemsToSend );

  bufferSize += nodeManager->packParentChildMapsSize( newNodesToSend );
  bufferSize += edgeManager->packParentChildMapsSize( newEdgesToSend );
  bufferSize += faceManager->packParentChildMapsSize( newFacesToSend );

  bufferSize += nodeManager->packSize( {}, newNodesToSend, 0 );
  bufferSize += edgeManager->packSize( {}, newEdgesToSend, 0 );
  bufferSize += faceManager->packSize( {}, newFacesToSend, 0 );
  bufferSize += elemManager->PackSize( {}, newElemsToSend );

  bufferSize += nodeManager->packUpDownMapsSize( modNodesToSend );
  bufferSize += edgeManager->packUpDownMapsSize( modEdgesToSend );
  bufferSize += faceManager->packUpDownMapsSize( modFacesToSend );
  bufferSize += elemManager->PackUpDownMapsSize( modElemsToSend );

  bufferSize += nodeManager->packParentChildMapsSize( modNodesToSend );
  bufferSize += edgeManager->packParentChildMapsSize( modEdgesToSend );
  bufferSize += faceManager->packParentChildMapsSize( modFacesToSend );


  neighbor->resizeSendBuffer( commID, bufferSize );

  buffer_type & sendBuffer = neighbor->sendBuffer( commID );
  buffer_unit_type * sendBufferPtr = sendBuffer.data();

  int packedSize = 0;

  packedSize += nodeManager->packGlobalMaps( sendBufferPtr, newNodesToSend, 0 );
  packedSize += edgeManager->packGlobalMaps( sendBufferPtr, newEdgesToSend, 0 );
  packedSize += faceManager->packGlobalMaps( sendBufferPtr, newFacesToSend, 0 );
  packedSize += elemManager->PackGlobalMaps( sendBufferPtr, newElemsToSend );

  packedSize += nodeManager->packUpDownMaps( sendBufferPtr, newNodesToSend );
  packedSize += edgeManager->packUpDownMaps( sendBufferPtr, newEdgesToSend );
  packedSize += faceManager->packUpDownMaps( sendBufferPtr, newFacesToSend );
  packedSize += elemManager->PackUpDownMaps( sendBufferPtr, newElemsToSend );

  packedSize += nodeManager->packParentChildMaps( sendBufferPtr, newNodesToSend );
  packedSize += edgeManager->packParentChildMaps( sendBufferPtr, newEdgesToSend );
  packedSize += faceManager->packParentChildMaps( sendBufferPtr, newFacesToSend );

  packedSize += nodeManager->pack( sendBufferPtr, {}, newNodesToSend, 0 );
  packedSize += edgeManager->pack( sendBufferPtr, {}, newEdgesToSend, 0 );
  packedSize += faceManager->pack( sendBufferPtr, {}, newFacesToSend, 0 );
  packedSize += elemManager->Pack( sendBufferPtr, {}, newElemsToSend );

  packedSize += nodeManager->packUpDownMaps( sendBufferPtr, modNodesToSend );
  packedSize += edgeManager->packUpDownMaps( sendBufferPtr, modEdgesToSend );
  packedSize += faceManager->packUpDownMaps( sendBufferPtr, modFacesToSend );
  packedSize += elemManager->PackUpDownMaps( sendBufferPtr, modElemsToSend );

  packedSize += nodeManager->packParentChildMaps( sendBufferPtr, modNodesToSend );
  packedSize += edgeManager->packParentChildMaps( sendBufferPtr, modEdgesToSend );
  packedSize += faceManager->packParentChildMaps( sendBufferPtr, modFacesToSend );


  GEOSX_ERROR_IF( bufferSize != packedSize, "Allocated Buffer Size is not equal to packed buffer size" );

  neighbor->mpiISendReceive( commID, MPI_COMM_GEOSX );
}

void ParallelTopologyChange::unpackNewModToGhosts( NeighborCommunicator * const neighbor,
                                                   int commID,
                                                   MeshLevel * const mesh,
                                                   ModifiedObjectLists & receivedObjects )
{
  int unpackedSize = 0;

  NodeManager * const nodeManager = mesh->getNodeManager();
  EdgeManager * const edgeManager = mesh->getEdgeManager();
  FaceManager * const faceManager = mesh->getFaceManager();
  ElementRegionManager * const elemManager = mesh->getElemManager();

  localIndex_array & nodeGhostsToRecv = nodeManager->getNeighborData( neighbor->neighborRank() ).ghostsToReceive();

  localIndex_array & edgeGhostsToRecv = edgeManager->getNeighborData( neighbor->neighborRank() ).ghostsToReceive();

  localIndex_array & faceGhostsToRecv = faceManager->getNeighborData( neighbor->neighborRank() ).ghostsToReceive();

  buffer_type const & receiveBuffer = neighbor->receiveBuffer( commID );
  buffer_unit_type const * receiveBufferPtr = receiveBuffer.data();

  localIndex_array newGhostNodes;
  localIndex_array newGhostEdges;
  localIndex_array newGhostFaces;

  localIndex_array modGhostNodes;
  localIndex_array modGhostEdges;
  localIndex_array modGhostFaces;

  ElementRegionManager::ElementReferenceAccessor< localIndex_array > newGhostElems;
  array1d< array1d< localIndex_array > > newGhostElemsData;
  ElementRegionManager::ElementReferenceAccessor< localIndex_array > modGhostElems;
  array1d< array1d< localIndex_array > > modGhostElemsData;
  newGhostElems.resize( elemManager->numRegions() );
  newGhostElemsData.resize( elemManager->numRegions() );
  modGhostElems.resize( elemManager->numRegions() );
  modGhostElemsData.resize( elemManager->numRegions() );
  for( localIndex er=0; er<elemManager->numRegions(); ++er )
  {
    ElementRegionBase * const elemRegion = elemManager->getRegion( er );
    newGhostElemsData[er].resize( elemRegion->numSubRegions() );
    newGhostElems[er].resize( elemRegion->numSubRegions() );
    modGhostElemsData[er].resize( elemRegion->numSubRegions() );
    modGhostElems[er].resize( elemRegion->numSubRegions() );
    for( localIndex esr=0; esr<elemRegion->numSubRegions(); ++esr )
    {
      newGhostElems[er][esr].set( newGhostElemsData[er][esr] );
      modGhostElems[er][esr].set( modGhostElemsData[er][esr] );
    }
  }


  unpackedSize += nodeManager->unpackGlobalMaps( receiveBufferPtr, newGhostNodes, 0 );
  unpackedSize += edgeManager->unpackGlobalMaps( receiveBufferPtr, newGhostEdges, 0 );
  unpackedSize += faceManager->unpackGlobalMaps( receiveBufferPtr, newGhostFaces, 0 );
  unpackedSize += elemManager->UnpackGlobalMaps( receiveBufferPtr, newGhostElems );

  unpackedSize += nodeManager->unpackUpDownMaps( receiveBufferPtr, newGhostNodes, true, true );
  unpackedSize += edgeManager->unpackUpDownMaps( receiveBufferPtr, newGhostEdges, true, true );
  unpackedSize += faceManager->unpackUpDownMaps( receiveBufferPtr, newGhostFaces, true, true );
  unpackedSize += elemManager->UnpackUpDownMaps( receiveBufferPtr, newGhostElems, true );

  unpackedSize += nodeManager->unpackParentChildMaps( receiveBufferPtr, newGhostNodes );
  unpackedSize += edgeManager->unpackParentChildMaps( receiveBufferPtr, newGhostEdges );
  unpackedSize += faceManager->unpackParentChildMaps( receiveBufferPtr, newGhostFaces );

  unpackedSize += nodeManager->unpack( receiveBufferPtr, newGhostNodes, 0 );
  unpackedSize += edgeManager->unpack( receiveBufferPtr, newGhostEdges, 0 );
  unpackedSize += faceManager->unpack( receiveBufferPtr, newGhostFaces, 0 );
  unpackedSize += elemManager->Unpack( receiveBufferPtr, newGhostElems );

  unpackedSize += nodeManager->unpackUpDownMaps( receiveBufferPtr, modGhostNodes, false, true );
  unpackedSize += edgeManager->unpackUpDownMaps( receiveBufferPtr, modGhostEdges, false, true );
  unpackedSize += faceManager->unpackUpDownMaps( receiveBufferPtr, modGhostFaces, false, true );
  unpackedSize += elemManager->UnpackUpDownMaps( receiveBufferPtr, modGhostElems, true );

  unpackedSize += nodeManager->unpackParentChildMaps( receiveBufferPtr, modGhostNodes );
  unpackedSize += edgeManager->unpackParentChildMaps( receiveBufferPtr, modGhostEdges );
  unpackedSize += faceManager->unpackParentChildMaps( receiveBufferPtr, modGhostFaces );

  if( newGhostNodes.size() > 0 )
  {
    nodeGhostsToRecv.move( LvArray::MemorySpace::CPU );
    for( localIndex a=0; a<newGhostNodes.size(); ++a )
    {
      nodeGhostsToRecv.emplace_back( newGhostNodes[a] );
    }
  }

  if( newGhostEdges.size() > 0 )
  {
    edgeGhostsToRecv.move( LvArray::MemorySpace::CPU );
    for( localIndex a=0; a<newGhostEdges.size(); ++a )
    {
      edgeGhostsToRecv.emplace_back( newGhostEdges[a] );
    }
  }

  if( newGhostFaces.size() > 0 )
  {
    faceGhostsToRecv.move( LvArray::MemorySpace::CPU );
    for( localIndex a=0; a<newGhostFaces.size(); ++a )
    {
      faceGhostsToRecv.emplace_back( newGhostFaces[a] );
    }
  }

  elemManager->forElementSubRegionsComplete< ElementSubRegionBase >(
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


    receivedObjects.modifiedElements[ { er, esr } ].insert( modGhostElemsData[er][esr].begin(),
                                                            modGhostElemsData[er][esr].end() );
  } );

  receivedObjects.newNodes.insert( newGhostNodes.begin(), newGhostNodes.end() );
  receivedObjects.modifiedNodes.insert( modGhostNodes.begin(), modGhostNodes.end() );
  receivedObjects.newEdges.insert( newGhostEdges.begin(), newGhostEdges.end() );
  receivedObjects.modifiedEdges.insert( modGhostEdges.begin(), modGhostEdges.end() );
  receivedObjects.newFaces.insert( newGhostFaces.begin(), newGhostFaces.end() );
  receivedObjects.modifiedFaces.insert( modGhostFaces.begin(), modGhostFaces.end() );

}

void ParallelTopologyChange::updateConnectorsToFaceElems( std::set< localIndex > const & newFaceElements,
                                                          FaceElementSubRegion const * const faceElemSubRegion,
                                                          EdgeManager * const edgeManager )
{
  ArrayOfArrays< localIndex > & connectorToElem = edgeManager->m_fractureConnectorEdgesToFaceElements;
  map< localIndex, localIndex > & edgesToConnectorEdges = edgeManager->m_edgesToFractureConnectorsEdges;
  array1d< localIndex > & connectorEdgesToEdges = edgeManager->m_fractureConnectorsEdgesToEdges;

  ArrayOfArraysView< localIndex const > const facesToEdges = faceElemSubRegion->edgeList().toViewConst();

  for( localIndex const & kfe : newFaceElements )
  {
    arraySlice1d< localIndex const > const faceToEdges = facesToEdges[kfe];
    for( localIndex ke=0; ke<faceToEdges.size(); ++ke )
    {
      localIndex const edgeIndex = faceToEdges[ke];

      auto connIter = edgesToConnectorEdges.find( edgeIndex );
      if( connIter==edgesToConnectorEdges.end() )
      {
        connectorToElem.appendArray( 0 );
        connectorEdgesToEdges.emplace_back( edgeIndex );
        edgesToConnectorEdges[edgeIndex] = connectorEdgesToEdges.size() - 1;
      }
      localIndex const connectorIndex = edgesToConnectorEdges.at( edgeIndex );

      localIndex const numExistingCells = connectorToElem.sizeOfArray( connectorIndex );
      bool cellExistsInMap = false;
      for( localIndex k=0; k<numExistingCells; ++k )
      {
        if( kfe == connectorToElem[connectorIndex][k] )
        {
          cellExistsInMap = true;
        }
      }
      if( !cellExistsInMap )
      {
        connectorToElem.resizeArray( connectorIndex, numExistingCells+1 );
        connectorToElem[connectorIndex][ numExistingCells ] = kfe;
        edgeManager->m_recalculateFractureConnectorEdges.insert( connectorIndex );
      }
    }
  }
}

} /* namespace geosx */
