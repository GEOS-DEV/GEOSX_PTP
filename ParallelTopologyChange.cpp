/**
 * @file ParallelTopologyChange.cpp
 */

#include "ParallelTopologyChange.hpp"
#include "common/GeosxMacros.hpp"
#include "common/TimingMacros.hpp"
#include "MPI_Communications/CommunicationTools.hpp"
#include "mesh/ElementRegionManager.hpp"


namespace geosx
{

using namespace dataRepository;

ParallelTopologyChange::ParallelTopologyChange()
{
}

ParallelTopologyChange::~ParallelTopologyChange()
{
}

void ParallelTopologyChange::SyncronizeTopologyChange( MeshLevel * const mesh,
                                                       array1d<NeighborCommunicator> & neighbors,
                                                       ModifiedObjectLists & modifiedObjects,
                                                       ModifiedObjectLists & receivedObjects )
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
  for( unsigned int neighborIndex=0 ; neighborIndex<neighbors.size() ; ++neighborIndex )
  {
    NeighborCommunicator & neighbor = neighbors[neighborIndex];

    PackNewAndModifiedObjectsToOwningRanks( &neighbor,
                                            mesh,
                                            modifiedObjects,
                                            commData.commID );

    neighbor.MPI_iSendReceiveBufferSizes( commData.commID,
                                          commData.mpiSizeSendBufferRequest[neighborIndex],
                                          commData.mpiSizeRecvBufferRequest[neighborIndex],
                                          MPI_COMM_GEOSX );

  }

  // send/recv the buffers
  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
  {
    int neighborIndex;
    MPI_Waitany( commData.size,
                 commData.mpiSizeRecvBufferRequest.data(),
                 &neighborIndex,
                 commData.mpiSizeRecvBufferStatus.data() );

    NeighborCommunicator& neighbor = neighbors[neighborIndex];

    neighbor.MPI_iSendReceiveBuffers( commData.commID,
                                      commData.mpiSendBufferRequest[neighborIndex],
                                      commData.mpiRecvBufferRequest[neighborIndex],
                                      MPI_COMM_GEOSX);
  }

  // unpack the buffers and get lists of the new objects.
  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
  {

    int neighborIndex;
    MPI_Waitany( commData.size,
                 commData.mpiRecvBufferRequest.data(),
                 &neighborIndex,
                 commData.mpiRecvBufferStatus.data() );

    NeighborCommunicator& neighbor = neighbors[neighborIndex];
    UnpackNewAndModifiedObjectsOnOwningRanks( &neighbor,
                                              mesh,
                                              commData.commID,
                                              receivedObjects );
  }

  nodeManager->inheritGhostRankFromParent( receivedObjects.newNodes );
  edgeManager->inheritGhostRankFromParent( receivedObjects.newEdges );
  faceManager->inheritGhostRankFromParent( receivedObjects.newFaces );

  elemManager->forElementSubRegionsComplete<FaceElementSubRegion>( [&]( localIndex const er,
                                                                        localIndex const esr,
                                                                        ElementRegionBase const * const GEOSX_UNUSED_ARG( elemRegion ),
                                                                        FaceElementSubRegion * const subRegion )
  {
    subRegion->inheritGhostRankFromParentFace( faceManager, receivedObjects.newElements[{er,esr}] );
  });

  MPI_Waitall( commData.size,
               commData.mpiSizeSendBufferRequest.data(),
               commData.mpiSizeSendBufferStatus.data() );

  MPI_Waitall( commData.size,
               commData.mpiSendBufferRequest.data(),
               commData.mpiSizeSendBufferStatus.data() );

  modifiedObjects.insert( receivedObjects );


  //************************************************************************************************
  // 3) now we need to send over:
  //   a) the new objects whose parents are ghosts on neighbors.
  //   b) the modified objects whose parents are ghosts on neighbors.



  MPI_iCommData commData2;
  commData2.resize(neighbors.size());
  for( unsigned int neighborIndex=0 ; neighborIndex<neighbors.size() ; ++neighborIndex )
  {
    NeighborCommunicator& neighbor = neighbors[neighborIndex];

    PackNewModifiedObjectsToGhosts( &neighbor,
                                    commData2.commID,
                                    mesh,
                                    modifiedObjects );

    neighbor.MPI_iSendReceiveBufferSizes( commData2.commID,
                                          commData2.mpiSizeSendBufferRequest[neighborIndex],
                                          commData2.mpiSizeRecvBufferRequest[neighborIndex],
                                          MPI_COMM_GEOSX );

  }

  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
  {
    int neighborIndex;
    MPI_Waitany( commData2.size,
                 commData2.mpiSizeRecvBufferRequest.data(),
                 &neighborIndex,
                 commData2.mpiSizeRecvBufferStatus.data() );

    NeighborCommunicator& neighbor = neighbors[neighborIndex];

    neighbor.MPI_iSendReceiveBuffers( commData2.commID,
                                      commData2.mpiSendBufferRequest[neighborIndex],
                                      commData2.mpiRecvBufferRequest[neighborIndex],
                                      MPI_COMM_GEOSX);
  }


  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
  {
    int neighborIndex;
    MPI_Waitany( commData2.size,
                 commData2.mpiRecvBufferRequest.data(),
                 &neighborIndex,
                 commData2.mpiRecvBufferStatus.data() );

    NeighborCommunicator& neighbor = neighbors[neighborIndex];
    UnpackNewModToGhosts( &neighbor, commData2.commID, mesh, receivedObjects );
  }

  modifiedObjects.insert( receivedObjects );

  nodeManager->FixUpDownMaps(false);
  edgeManager->FixUpDownMaps(false);
  faceManager->FixUpDownMaps(false);


  for( localIndex er=0 ; er<elemManager->numRegions() ; ++er )
  {
    ElementRegionBase * const elemRegion = elemManager->GetRegion(er);
    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      ElementSubRegionBase * const subRegion = elemRegion->GetSubRegion(esr);
      subRegion->FixUpDownMaps(false);
    }
  }

  elemManager->forElementSubRegionsComplete<FaceElementSubRegion>([&]( localIndex const er,
                                                                       localIndex const esr,
                                                                       ElementRegionBase const * const GEOSX_UNUSED_ARG( elemRegion ),
                                                                       FaceElementSubRegion const * const subRegion )
  {
    updateConnectorsToFaceElems( receivedObjects.newElements.at({er,esr}),
                                 subRegion,
                                 edgeManager );
  });


  std::set<localIndex> allTouchedNodes;
  allTouchedNodes.insert( modifiedObjects.newNodes.begin(), modifiedObjects.newNodes.end() );
  allTouchedNodes.insert( modifiedObjects.modifiedNodes.begin(), modifiedObjects.modifiedNodes.end() );
  nodeManager->depopulateUpMaps( allTouchedNodes,
                                 edgeManager->nodeList(),
                                 faceManager->nodeList(),
                                 *elemManager );

  std::set<localIndex> allTouchedEdges;
  allTouchedEdges.insert( modifiedObjects.newEdges.begin(), modifiedObjects.newEdges.end() );
  allTouchedEdges.insert( modifiedObjects.modifiedEdges.begin(), modifiedObjects.modifiedEdges.end() );
  edgeManager->depopulateUpMaps( allTouchedEdges,
                                 faceManager->edgeList() );

  std::set<localIndex> allTouchedFaces;
  allTouchedFaces.insert( modifiedObjects.newFaces.begin(), modifiedObjects.newFaces.end() );
  allTouchedFaces.insert( modifiedObjects.modifiedFaces.begin(), modifiedObjects.modifiedFaces.end() );
  faceManager->depopulateUpMaps( allTouchedFaces,
                                 *elemManager );

  nodeManager->enforceStateFieldConsistencyPostTopologyChange(modifiedObjects.modifiedNodes);
  edgeManager->enforceStateFieldConsistencyPostTopologyChange(modifiedObjects.modifiedEdges);
  faceManager->enforceStateFieldConsistencyPostTopologyChange(modifiedObjects.modifiedFaces);


}


void
ParallelTopologyChange::
PackNewAndModifiedObjectsToOwningRanks( NeighborCommunicator * const neighbor,
                                        MeshLevel * const meshLevel,
                                        ModifiedObjectLists const & modifiedObjects,
                                        int const commID )
{
  int bufferSize = 0;

  NodeManager & nodeManager = *(meshLevel->getNodeManager());
  EdgeManager & edgeManager = *(meshLevel->getEdgeManager());
  FaceManager & faceManager = *(meshLevel->getFaceManager());
  ElementRegionManager & elemManager = *(meshLevel->getElemManager() );

  array1d<integer> const & nodeGhostRank = nodeManager.GhostRank();
  array1d<integer> const & edgeGhostRank = edgeManager.GhostRank();
  array1d<integer> const & faceGhostRank = faceManager.GhostRank();

  arrayView1d<localIndex> const &
  parentNodeIndices = nodeManager.getReference<array1d<localIndex>>( nodeManager.viewKeys.parentIndex );

  arrayView1d<localIndex> const &
  parentEdgeIndices = edgeManager.getReference<array1d<localIndex>>( edgeManager.viewKeys.parentIndex );

  arrayView1d<localIndex> const &
  parentFaceIndices = faceManager.getReference<array1d<localIndex>>( faceManager.viewKeys.parentIndex );

  int const neighborRank = neighbor->NeighborRank();

  array1d<localIndex> newNodePackListArray(modifiedObjects.newNodes.size());
  {
    localIndex a=0 ;
    for( auto const index : modifiedObjects.newNodes )
    {
      localIndex const parentNodeIndex = ObjectManagerBase::GetParentRecusive( parentNodeIndices, index );
      if( nodeGhostRank[parentNodeIndex] == neighborRank )
      {
        newNodePackListArray[a] = index;
        ++a;
      }
    }
    newNodePackListArray.resize(a);
  }
  array1d<localIndex> modNodePackListArray(modifiedObjects.modifiedNodes.size());
  {
    localIndex a=0 ;
    for( auto const index : modifiedObjects.modifiedNodes )
    {
      localIndex const parentNodeIndex = ObjectManagerBase::GetParentRecusive( parentNodeIndices, index );
      if( nodeGhostRank[parentNodeIndex] == neighborRank )
      {
        modNodePackListArray[a] = index;
        ++a;
      }
    }
    modNodePackListArray.resize(a);
  }


  array1d<localIndex> newEdgePackListArray(modifiedObjects.newEdges.size());
  {
    localIndex a=0 ;
    for( auto const index : modifiedObjects.newEdges )
    {
      localIndex const parentIndex = ObjectManagerBase::GetParentRecusive( parentEdgeIndices, index );
      if( edgeGhostRank[parentIndex] == neighborRank )
      {
        newEdgePackListArray[a] = index;
        ++a;
      }
    }
    newEdgePackListArray.resize(a);
  }
  array1d<localIndex> modEdgePackListArray(modifiedObjects.modifiedEdges.size());
  {
    localIndex a=0 ;
    for( auto const index : modifiedObjects.modifiedEdges )
    {
      localIndex const parentIndex = ObjectManagerBase::GetParentRecusive( parentEdgeIndices, index );
      if( edgeGhostRank[parentIndex] == neighborRank )
      {
        modEdgePackListArray[a] = index;
        ++a;
      }
    }
    modEdgePackListArray.resize(a);
  }


  array1d<localIndex> newFacePackListArray(modifiedObjects.newFaces.size());
  {
    localIndex a=0 ;
    for( auto const index : modifiedObjects.newFaces )
    {
      localIndex const parentIndex = ObjectManagerBase::GetParentRecusive( parentFaceIndices, index );
      if( faceGhostRank[parentIndex] == neighborRank )
      {
        newFacePackListArray[a] = index;
        ++a;
      }
    }
    newFacePackListArray.resize(a);
  }
  array1d<localIndex> modFacePackListArray(modifiedObjects.modifiedFaces.size());
  {
    localIndex a=0 ;
    for( auto const index : modifiedObjects.modifiedFaces )
    {
      localIndex const parentIndex = ObjectManagerBase::GetParentRecusive( parentFaceIndices, index );
      if( faceGhostRank[parentIndex] == neighborRank )
      {
        modFacePackListArray[a] = index;
        ++a;
      }
    }
    modFacePackListArray.resize(a);
  }


  ElementRegionManager::ElementViewAccessor< arrayView1d<localIndex> > newElemPackList;
  array1d<array1d<localIndex_array> > newElemData;
  ElementRegionManager::ElementReferenceAccessor< localIndex_array > modElemPackList;
  array1d<array1d<localIndex_array> > modElemData;
  newElemPackList.resize(elemManager.numRegions());
  newElemData.resize(elemManager.numRegions());
  modElemPackList.resize(elemManager.numRegions());
  modElemData.resize(elemManager.numRegions());
  for( localIndex er=0 ; er<elemManager.numRegions() ; ++er )
  {
    ElementRegionBase * const elemRegion = elemManager.GetRegion(er);
    newElemPackList[er].resize(elemRegion->numSubRegions());
    newElemData[er].resize(elemRegion->numSubRegions());
    modElemPackList[er].resize(elemRegion->numSubRegions());
    modElemData[er].resize(elemRegion->numSubRegions());
    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      ElementSubRegionBase * const subRegion = elemRegion->GetSubRegion(esr);
      integer_array const & subRegionGhostRank = subRegion->GhostRank();
      if( modifiedObjects.modifiedElements.count({er,esr}) > 0 )
      {
        std::set<localIndex> const & elemList = modifiedObjects.modifiedElements.at({er,esr});

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
        modElemData[er][esr].resize(a);
      }

      if( modifiedObjects.newElements.count({er,esr}) > 0 )
      {
        std::set<localIndex> const & elemList = modifiedObjects.newElements.at({er,esr});

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
        newElemData[er][esr].resize(a);
      }

      newElemPackList[er][esr] = newElemData[er][esr] ;
      modElemPackList[er][esr].set( modElemData[er][esr] );
    }
  }








  bufferSize += nodeManager.PackGlobalMapsSize( newNodePackListArray, 0 );
  bufferSize += edgeManager.PackGlobalMapsSize( newEdgePackListArray, 0 );
  bufferSize += faceManager.PackGlobalMapsSize( newFacePackListArray, 0 );
  bufferSize += elemManager.PackGlobalMapsSize( newElemPackList );

  bufferSize += nodeManager.PackUpDownMapsSize( newNodePackListArray );
  bufferSize += edgeManager.PackUpDownMapsSize( newEdgePackListArray );
  bufferSize += faceManager.PackUpDownMapsSize( newFacePackListArray );
  bufferSize += elemManager.PackUpDownMapsSize( newElemPackList );

  bufferSize += nodeManager.PackParentChildMapsSize( newNodePackListArray );
  bufferSize += edgeManager.PackParentChildMapsSize( newEdgePackListArray );
  bufferSize += faceManager.PackParentChildMapsSize( newFacePackListArray );

  bufferSize += nodeManager.PackSize( {}, newNodePackListArray, 0 );
  bufferSize += edgeManager.PackSize( {}, newEdgePackListArray, 0 );
  bufferSize += faceManager.PackSize( {}, newFacePackListArray, 0 );
  bufferSize += elemManager.PackSize( {}, newElemPackList );

  bufferSize += nodeManager.PackUpDownMapsSize( modNodePackListArray );
  bufferSize += edgeManager.PackUpDownMapsSize( modEdgePackListArray );
  bufferSize += faceManager.PackUpDownMapsSize( modFacePackListArray );
  bufferSize += elemManager.PackUpDownMapsSize( modElemPackList );

  bufferSize += nodeManager.PackParentChildMapsSize( modNodePackListArray );
  bufferSize += edgeManager.PackParentChildMapsSize( modEdgePackListArray );
  bufferSize += faceManager.PackParentChildMapsSize( modFacePackListArray );

  bufferSize += nodeManager.PackSize( {}, modNodePackListArray, 0 );
  bufferSize += edgeManager.PackSize( {}, modEdgePackListArray, 0 );
  bufferSize += faceManager.PackSize( {}, modFacePackListArray, 0 );


  neighbor->resizeSendBuffer( commID, bufferSize );

  buffer_type & sendBuffer = neighbor->SendBuffer( commID );
  buffer_unit_type * sendBufferPtr = sendBuffer.data();

  int packedSize = 0;

  packedSize += nodeManager.PackGlobalMaps( sendBufferPtr, newNodePackListArray, 0 );
  packedSize += edgeManager.PackGlobalMaps( sendBufferPtr, newEdgePackListArray, 0 );
  packedSize += faceManager.PackGlobalMaps( sendBufferPtr, newFacePackListArray, 0 );
  packedSize += elemManager.PackGlobalMaps( sendBufferPtr, newElemPackList );

  packedSize += nodeManager.PackUpDownMaps( sendBufferPtr, newNodePackListArray );
  packedSize += edgeManager.PackUpDownMaps( sendBufferPtr, newEdgePackListArray );
  packedSize += faceManager.PackUpDownMaps( sendBufferPtr, newFacePackListArray );
  packedSize += elemManager.PackUpDownMaps( sendBufferPtr, newElemPackList );

  packedSize += nodeManager.PackParentChildMaps( sendBufferPtr, newNodePackListArray );
  packedSize += edgeManager.PackParentChildMaps( sendBufferPtr, newEdgePackListArray );
  packedSize += faceManager.PackParentChildMaps( sendBufferPtr, newFacePackListArray );

  packedSize += nodeManager.Pack( sendBufferPtr, {}, newNodePackListArray, 0 );
  packedSize += edgeManager.Pack( sendBufferPtr, {}, newEdgePackListArray, 0 );
  packedSize += faceManager.Pack( sendBufferPtr, {}, newFacePackListArray, 0 );
  packedSize += elemManager.Pack( sendBufferPtr, {}, newElemPackList );

  packedSize += nodeManager.PackUpDownMaps( sendBufferPtr, modNodePackListArray );
  packedSize += edgeManager.PackUpDownMaps( sendBufferPtr, modEdgePackListArray );
  packedSize += faceManager.PackUpDownMaps( sendBufferPtr, modFacePackListArray );
  packedSize += elemManager.PackUpDownMaps( sendBufferPtr, modElemPackList );

  packedSize += nodeManager.PackParentChildMaps( sendBufferPtr, modNodePackListArray );
  packedSize += edgeManager.PackParentChildMaps( sendBufferPtr, modEdgePackListArray );
  packedSize += faceManager.PackParentChildMaps( sendBufferPtr, modFacePackListArray );

  packedSize += nodeManager.Pack( sendBufferPtr, {}, modNodePackListArray, 0 );
  packedSize += edgeManager.Pack( sendBufferPtr, {}, modEdgePackListArray, 0 );
  packedSize += faceManager.Pack( sendBufferPtr, {}, modFacePackListArray, 0 );


  GEOS_ERROR_IF( bufferSize != packedSize,
                 "Allocated Buffer Size ("<<bufferSize<<") is not equal to packed buffer size("<<packedSize<<")" );


}

localIndex
ParallelTopologyChange::
UnpackNewAndModifiedObjectsOnOwningRanks( NeighborCommunicator * const neighbor,
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





  buffer_type const & receiveBuffer = neighbor->ReceiveBuffer( commID );
  buffer_unit_type const * receiveBufferPtr = receiveBuffer.data();
  localIndex_array newLocalNodes, modifiedLocalNodes;
  localIndex_array newLocalEdges, modifiedLocalEdges;
  localIndex_array newLocalFaces, modifiedLocalFaces;

  ElementRegionManager::ElementReferenceAccessor< array1d<localIndex> > newLocalElements;
  array1d<array1d<localIndex_array> > newLocalElementsData;

  ElementRegionManager::ElementReferenceAccessor< localIndex_array > modifiedLocalElements;
  array1d<array1d<localIndex_array> > modifiedLocalElementsData;

  newLocalElements.resize(elemManager->numRegions());
  newLocalElementsData.resize(elemManager->numRegions());
  modifiedLocalElements.resize(elemManager->numRegions());
  modifiedLocalElementsData.resize(elemManager->numRegions());
  for( localIndex er=0 ; er<elemManager->numRegions() ; ++er )
  {
    ElementRegionBase * const elemRegion = elemManager->GetRegion(er);
    newLocalElements[er].resize(elemRegion->numSubRegions());
    newLocalElementsData[er].resize(elemRegion->numSubRegions());
    modifiedLocalElements[er].resize(elemRegion->numSubRegions());
    modifiedLocalElementsData[er].resize(elemRegion->numSubRegions());
    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      newLocalElements[er][esr].set(newLocalElementsData[er][esr]);
      modifiedLocalElements[er][esr].set( modifiedLocalElementsData[er][esr] );
    }
  }





  int unpackedSize = 0;
  unpackedSize += nodeManager->UnpackGlobalMaps( receiveBufferPtr, newLocalNodes, 0 );
  unpackedSize += edgeManager->UnpackGlobalMaps( receiveBufferPtr, newLocalEdges, 0 );
  unpackedSize += faceManager->UnpackGlobalMaps( receiveBufferPtr, newLocalFaces, 0 );
  unpackedSize += elemManager->UnpackGlobalMaps( receiveBufferPtr, newLocalElements );

  unpackedSize += nodeManager->UnpackUpDownMaps( receiveBufferPtr, newLocalNodes, true, true );
  unpackedSize += edgeManager->UnpackUpDownMaps( receiveBufferPtr, newLocalEdges, true, true );
  unpackedSize += faceManager->UnpackUpDownMaps( receiveBufferPtr, newLocalFaces, true, true );
  unpackedSize += elemManager->UnpackUpDownMaps( receiveBufferPtr, newLocalElements, true );

  unpackedSize += nodeManager->UnpackParentChildMaps( receiveBufferPtr, newLocalNodes );
  unpackedSize += edgeManager->UnpackParentChildMaps( receiveBufferPtr, newLocalEdges );
  unpackedSize += faceManager->UnpackParentChildMaps( receiveBufferPtr, newLocalFaces );

  unpackedSize += nodeManager->Unpack( receiveBufferPtr, newLocalNodes, 0 );
  unpackedSize += edgeManager->Unpack( receiveBufferPtr, newLocalEdges, 0 );
  unpackedSize += faceManager->Unpack( receiveBufferPtr, newLocalFaces, 0 );
  unpackedSize += elemManager->Unpack( receiveBufferPtr, newLocalElements );

  unpackedSize += nodeManager->UnpackUpDownMaps( receiveBufferPtr, modifiedLocalNodes, false, true );
  unpackedSize += edgeManager->UnpackUpDownMaps( receiveBufferPtr, modifiedLocalEdges, false, true );
  unpackedSize += faceManager->UnpackUpDownMaps( receiveBufferPtr, modifiedLocalFaces, false, true );
  unpackedSize += elemManager->UnpackUpDownMaps( receiveBufferPtr, modifiedLocalElements, true );

  unpackedSize += nodeManager->UnpackParentChildMaps( receiveBufferPtr, modifiedLocalNodes );
  unpackedSize += edgeManager->UnpackParentChildMaps( receiveBufferPtr, modifiedLocalEdges );
  unpackedSize += faceManager->UnpackParentChildMaps( receiveBufferPtr, modifiedLocalFaces );

  unpackedSize += nodeManager->Unpack( receiveBufferPtr, modifiedLocalNodes, 0 );
  unpackedSize += edgeManager->Unpack( receiveBufferPtr, modifiedLocalEdges, 0 );
  unpackedSize += faceManager->Unpack( receiveBufferPtr, modifiedLocalFaces, 0 );
//    unpackedSize += elemManager->Unpack( receiveBufferPtr, modifiedElements );



  std::set<localIndex> & allNewNodes      = receivedObjects.newNodes;
  std::set<localIndex> & allModifiedNodes = receivedObjects.modifiedNodes;
  std::set<localIndex> & allNewEdges      = receivedObjects.newEdges;
  std::set<localIndex> & allModifiedEdges = receivedObjects.modifiedEdges;
  std::set<localIndex> & allNewFaces      = receivedObjects.newFaces;
  std::set<localIndex> & allModifiedFaces = receivedObjects.modifiedFaces;
  map< std::pair<localIndex, localIndex>, std::set<localIndex> > & allNewElements = receivedObjects.newElements;
  map< std::pair<localIndex, localIndex>, std::set<localIndex> > & allModifiedElements = receivedObjects.modifiedElements;

  allNewNodes.insert( newLocalNodes.begin(), newLocalNodes.end() );
  allModifiedNodes.insert( modifiedLocalNodes.begin(), modifiedLocalNodes.end() );

  allNewEdges.insert( newLocalEdges.begin(), newLocalEdges.end() );
  allModifiedEdges.insert( modifiedLocalEdges.begin(), modifiedLocalEdges.end() );

  allNewFaces.insert( newLocalFaces.begin(), newLocalFaces.end() );
  allModifiedFaces.insert( modifiedLocalFaces.begin(), modifiedLocalFaces.end() );

  for( localIndex er=0 ; er<elemManager->numRegions() ; ++er )
  {
    ElementRegionBase * const elemRegion = elemManager->GetRegion(er);
    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      allNewElements[{er,esr}].insert( newLocalElements[er][esr].get().begin(),
                                       newLocalElements[er][esr].get().end() );
      allModifiedElements[{er,esr}].insert( modifiedLocalElements[er][esr].get().begin(),
                                            modifiedLocalElements[er][esr].get().end() );
    }
  }

  return unpackedSize;
}


static void FilterNewObjectsForPackToGhosts( std::set<localIndex> const & objectList,
                                             localIndex_array const & parentIndices,
                                             localIndex_array & ghostsToSend,
                                             localIndex_array & objectsToSend )
{
  //TODO this needs to be inverted since the ghostToSend list should be much longer....
  // and the objectList is a searchable set.
  for( auto const index : objectList )
  {
    localIndex const parentIndex = parentIndices[index];
    for( localIndex a=0 ; a<ghostsToSend.size() ; ++a )
    {
      if( ghostsToSend[a]==parentIndex )
      {
        objectsToSend.push_back( index );
        ghostsToSend.push_back(index);
        break;
      }
    }
  }
}

static void FilterModObjectsForPackToGhosts( std::set<localIndex> const & objectList,
                                             localIndex_array const & ghostsToSend,
                                             localIndex_array & objectsToSend )
{
  for( localIndex a=0 ; a<ghostsToSend.size() ; ++a )
  {
    if( objectList.count(ghostsToSend[a]) > 0)
    {
      objectsToSend.push_back( ghostsToSend[a] );
    }
  }
}

void ParallelTopologyChange::PackNewModifiedObjectsToGhosts( NeighborCommunicator * const neighbor,
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
  ElementRegionManager::ElementViewAccessor< arrayView1d<localIndex> > newElemsToSend;
  array1d< array1d <localIndex_array> > newElemsToSendData;

  localIndex_array modNodesToSend;
  localIndex_array modEdgesToSend;
  localIndex_array modFacesToSend;
  ElementRegionManager::ElementReferenceAccessor<localIndex_array> modElemsToSend;
  array1d< array1d <localIndex_array> > modElemsToSendData;

  Group *
  nodeNeighborData = nodeManager->GetGroup( nodeManager->groupKeys.neighborData )->
                                  GetGroup( std::to_string( neighbor->NeighborRank() ) );

  Group *
  edgeNeighborData = edgeManager->GetGroup( edgeManager->groupKeys.neighborData )->
                                  GetGroup( std::to_string( neighbor->NeighborRank() ) );

  Group *
  faceNeighborData = faceManager->GetGroup( faceManager->groupKeys.neighborData )->
                                  GetGroup( std::to_string( neighbor->NeighborRank() ) );

  localIndex_array &
  nodeGhostsToSend = nodeNeighborData->getReference<localIndex_array>( nodeManager->viewKeys.ghostsToSend );

  localIndex_array &
  edgeGhostsToSend = edgeNeighborData->getReference<localIndex_array>( nodeManager->viewKeys.ghostsToSend );

  localIndex_array &
  faceGhostsToSend = faceNeighborData->getReference<localIndex_array>( faceManager->viewKeys.ghostsToSend );

  ElementRegionManager::ElementReferenceAccessor<localIndex_array>
  elementGhostToSend =
     elemManager->ConstructReferenceAccessor<localIndex_array>( ObjectManagerBase::
                                                          viewKeyStruct::
                                                          ghostsToSendString,
                                                          std::to_string( neighbor->NeighborRank() ) );


  localIndex_array const &
  nodalParentIndices = nodeManager->getReference<localIndex_array>( nodeManager->
                                                                     m_ObjectManagerBaseViewKeys.
                                                                     parentIndex );

  localIndex_array const &
  edgeParentIndices = edgeManager->getReference<localIndex_array>( edgeManager->
                                                                    m_ObjectManagerBaseViewKeys.
                                                                    parentIndex );

  localIndex_array const &
  faceParentIndices = faceManager->getReference<localIndex_array>( faceManager->
                                                                    m_ObjectManagerBaseViewKeys.
                                                                    parentIndex );

  FilterNewObjectsForPackToGhosts( receivedObjects.newNodes, nodalParentIndices, nodeGhostsToSend, newNodesToSend );
  FilterModObjectsForPackToGhosts( receivedObjects.modifiedNodes, nodeGhostsToSend, modNodesToSend );

  FilterNewObjectsForPackToGhosts( receivedObjects.newEdges, edgeParentIndices, edgeGhostsToSend, newEdgesToSend );
  FilterModObjectsForPackToGhosts( receivedObjects.modifiedEdges, edgeGhostsToSend, modEdgesToSend );

  FilterNewObjectsForPackToGhosts( receivedObjects.newFaces, faceParentIndices, faceGhostsToSend, newFacesToSend );
  FilterModObjectsForPackToGhosts( receivedObjects.modifiedFaces, faceGhostsToSend, modFacesToSend );

  set<localIndex> faceGhostsToSendSet;
  for( localIndex const & kf : faceGhostsToSend )
  {
    faceGhostsToSendSet.insert(kf);
  }

  newElemsToSendData.resize( elemManager->numRegions() );
  newElemsToSend.resize( elemManager->numRegions() );
  modElemsToSendData.resize( elemManager->numRegions() );
  modElemsToSend.resize( elemManager->numRegions() );
  for( localIndex er=0 ; er<elemManager->numRegions() ; ++er )
  {
    ElementRegionBase * const elemRegion = elemManager->GetRegion(er);
    newElemsToSendData[er].resize( elemRegion->numSubRegions() );
    newElemsToSend[er].resize( elemRegion->numSubRegions() );
    modElemsToSendData[er].resize( elemRegion->numSubRegions() );
    modElemsToSend[er].resize( elemRegion->numSubRegions() );

    elemRegion->forElementSubRegionsIndex<FaceElementSubRegion>( [&]( localIndex const esr,
                                                                      FaceElementSubRegion const * const subRegion )
    {
      FaceElementSubRegion::FaceMapType const & faceList = subRegion->faceList();
      for( localIndex const & k : receivedObjects.newElements.at({er,esr}) )
      {
        if( faceGhostsToSendSet.count( faceList(k,0) ) )
        {
          newElemsToSendData[er][esr].push_back(k);
          elementGhostToSend[er][esr].get().push_back(k);
        }
      }
      newElemsToSend[er][esr] = newElemsToSendData[er][esr] ;
    });
    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      modElemsToSend[er][esr].set( modElemsToSendData[er][esr] );
      for( localIndex a=0 ; a<elementGhostToSend[er][esr].get().size() ; ++a )
      {
        if( receivedObjects.modifiedElements.at({er,esr}).count( elementGhostToSend[er][esr][a] ) > 0 )
        {
          modElemsToSendData[er][esr].push_back(elementGhostToSend[er][esr][a]);
        }
      }
    }
  }


  int bufferSize = 0;

  bufferSize += nodeManager->PackGlobalMapsSize( newNodesToSend, 0 );
  bufferSize += edgeManager->PackGlobalMapsSize( newEdgesToSend, 0 );
  bufferSize += faceManager->PackGlobalMapsSize( newFacesToSend, 0 );
  bufferSize += elemManager->PackGlobalMapsSize( newElemsToSend );

  bufferSize += nodeManager->PackUpDownMapsSize( newNodesToSend );
  bufferSize += edgeManager->PackUpDownMapsSize( newEdgesToSend );
  bufferSize += faceManager->PackUpDownMapsSize( newFacesToSend );
  bufferSize += elemManager->PackUpDownMapsSize( newElemsToSend );

  bufferSize += nodeManager->PackParentChildMapsSize( newNodesToSend );
  bufferSize += edgeManager->PackParentChildMapsSize( newEdgesToSend );
  bufferSize += faceManager->PackParentChildMapsSize( newFacesToSend );

  bufferSize += nodeManager->PackSize( {}, newNodesToSend, 0 );
  bufferSize += edgeManager->PackSize( {}, newEdgesToSend, 0 );
  bufferSize += faceManager->PackSize( {}, newFacesToSend, 0 );
  bufferSize += elemManager->PackSize( {}, newElemsToSend );

  bufferSize += nodeManager->PackUpDownMapsSize( modNodesToSend );
  bufferSize += edgeManager->PackUpDownMapsSize( modEdgesToSend );
  bufferSize += faceManager->PackUpDownMapsSize( modFacesToSend );
  bufferSize += elemManager->PackUpDownMapsSize( modElemsToSend );

  bufferSize += nodeManager->PackParentChildMapsSize( modNodesToSend );
  bufferSize += edgeManager->PackParentChildMapsSize( modEdgesToSend );
  bufferSize += faceManager->PackParentChildMapsSize( modFacesToSend );


  neighbor->resizeSendBuffer( commID, bufferSize );

  buffer_type & sendBuffer = neighbor->SendBuffer( commID );
  buffer_unit_type * sendBufferPtr = sendBuffer.data();

  int packedSize = 0;

  packedSize += nodeManager->PackGlobalMaps( sendBufferPtr, newNodesToSend, 0 );
  packedSize += edgeManager->PackGlobalMaps( sendBufferPtr, newEdgesToSend, 0 );
  packedSize += faceManager->PackGlobalMaps( sendBufferPtr, newFacesToSend, 0 );
  packedSize += elemManager->PackGlobalMaps( sendBufferPtr, newElemsToSend );

  packedSize += nodeManager->PackUpDownMaps( sendBufferPtr, newNodesToSend );
  packedSize += edgeManager->PackUpDownMaps( sendBufferPtr, newEdgesToSend );
  packedSize += faceManager->PackUpDownMaps( sendBufferPtr, newFacesToSend );
  packedSize += elemManager->PackUpDownMaps( sendBufferPtr, newElemsToSend );

  packedSize += nodeManager->PackParentChildMaps( sendBufferPtr, newNodesToSend );
  packedSize += edgeManager->PackParentChildMaps( sendBufferPtr, newEdgesToSend );
  packedSize += faceManager->PackParentChildMaps( sendBufferPtr, newFacesToSend );

  packedSize += nodeManager->Pack( sendBufferPtr, {}, newNodesToSend, 0 );
  packedSize += edgeManager->Pack( sendBufferPtr, {}, newEdgesToSend, 0 );
  packedSize += faceManager->Pack( sendBufferPtr, {}, newFacesToSend, 0 );
  packedSize += elemManager->Pack( sendBufferPtr, {}, newElemsToSend );

  packedSize += nodeManager->PackUpDownMaps( sendBufferPtr, modNodesToSend );
  packedSize += edgeManager->PackUpDownMaps( sendBufferPtr, modEdgesToSend );
  packedSize += faceManager->PackUpDownMaps( sendBufferPtr, modFacesToSend );
  packedSize += elemManager->PackUpDownMaps( sendBufferPtr, modElemsToSend );

  packedSize += nodeManager->PackParentChildMaps( sendBufferPtr, modNodesToSend );
  packedSize += edgeManager->PackParentChildMaps( sendBufferPtr, modEdgesToSend );
  packedSize += faceManager->PackParentChildMaps( sendBufferPtr, modFacesToSend );


  GEOS_ERROR_IF( bufferSize != packedSize, "Allocated Buffer Size is not equal to packed buffer size" );

  neighbor->MPI_iSendReceive( commID, MPI_COMM_GEOSX );


}

void ParallelTopologyChange::UnpackNewModToGhosts( NeighborCommunicator * const neighbor,
                                                   int commID,
                                                   MeshLevel * const mesh,
                                                   ModifiedObjectLists & receivedObjects )
{
  int unpackedSize = 0;

  NodeManager * const nodeManager = mesh->getNodeManager();
  EdgeManager * const edgeManager = mesh->getEdgeManager();
  FaceManager * const faceManager = mesh->getFaceManager();
  ElementRegionManager * const elemManager = mesh->getElemManager();


  Group *
   nodeNeighborData = nodeManager->GetGroup( nodeManager->groupKeys.neighborData )->
                      GetGroup( std::to_string( neighbor->NeighborRank() ) );

   Group *
   edgeNeighborData = edgeManager->GetGroup( edgeManager->groupKeys.neighborData )->
                      GetGroup( std::to_string( neighbor->NeighborRank() ) );

   Group *
   faceNeighborData = faceManager->GetGroup( faceManager->groupKeys.neighborData )->
                      GetGroup( std::to_string( neighbor->NeighborRank() ) );

   localIndex_array &
   nodeGhostsToRecv = nodeNeighborData->getReference<localIndex_array>( nodeManager->viewKeys.ghostsToReceive );

   localIndex_array &
   edgeGhostsToRecv = edgeNeighborData->getReference<localIndex_array>( nodeManager->viewKeys.ghostsToReceive );

   localIndex_array &
   faceGhostsToRecv = faceNeighborData->getReference<localIndex_array>( faceManager->viewKeys.ghostsToReceive );

   ElementRegionManager::ElementReferenceAccessor<localIndex_array>
   elementGhostToReceive =
     elemManager->ConstructReferenceAccessor<localIndex_array>( ObjectManagerBase::
                                                          viewKeyStruct::
                                                          ghostsToReceiveString,
                                                          std::to_string( neighbor->NeighborRank() ) );



  buffer_type const & receiveBuffer = neighbor->ReceiveBuffer( commID );
  buffer_unit_type const * receiveBufferPtr = receiveBuffer.data();

  localIndex_array newGhostNodes;
  localIndex_array newGhostEdges;
  localIndex_array newGhostFaces;

  localIndex_array modGhostNodes;
  localIndex_array modGhostEdges;
  localIndex_array modGhostFaces;

  ElementRegionManager::ElementReferenceAccessor<localIndex_array> newGhostElems;
  array1d< array1d <localIndex_array> > newGhostElemsData;
  ElementRegionManager::ElementReferenceAccessor<localIndex_array> modGhostElems;
  array1d< array1d <localIndex_array> > modGhostElemsData;
  newGhostElems.resize( elemManager->numRegions() );
  newGhostElemsData.resize( elemManager->numRegions() );
  modGhostElems.resize( elemManager->numRegions() );
  modGhostElemsData.resize( elemManager->numRegions() );
  for( localIndex er=0 ; er<elemManager->numRegions() ; ++er )
  {
    ElementRegionBase * const elemRegion = elemManager->GetRegion(er);
    newGhostElemsData[er].resize( elemRegion->numSubRegions() );
    newGhostElems[er].resize( elemRegion->numSubRegions() );
    modGhostElemsData[er].resize( elemRegion->numSubRegions() );
    modGhostElems[er].resize( elemRegion->numSubRegions() );
    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      newGhostElems[er][esr].set( newGhostElemsData[er][esr] );
      modGhostElems[er][esr].set( modGhostElemsData[er][esr] );
    }
  }


  unpackedSize += nodeManager->UnpackGlobalMaps( receiveBufferPtr, newGhostNodes, 0 );
  unpackedSize += edgeManager->UnpackGlobalMaps( receiveBufferPtr, newGhostEdges, 0 );
  unpackedSize += faceManager->UnpackGlobalMaps( receiveBufferPtr, newGhostFaces, 0 );
  unpackedSize += elemManager->UnpackGlobalMaps( receiveBufferPtr, newGhostElems );

  unpackedSize += nodeManager->UnpackUpDownMaps( receiveBufferPtr, newGhostNodes, true, true );
  unpackedSize += edgeManager->UnpackUpDownMaps( receiveBufferPtr, newGhostEdges, true, true );
  unpackedSize += faceManager->UnpackUpDownMaps( receiveBufferPtr, newGhostFaces, true, true );
  unpackedSize += elemManager->UnpackUpDownMaps( receiveBufferPtr, newGhostElems, true );

  unpackedSize += nodeManager->UnpackParentChildMaps( receiveBufferPtr, newGhostNodes );
  unpackedSize += edgeManager->UnpackParentChildMaps( receiveBufferPtr, newGhostEdges );
  unpackedSize += faceManager->UnpackParentChildMaps( receiveBufferPtr, newGhostFaces );

  unpackedSize += nodeManager->Unpack( receiveBufferPtr, newGhostNodes, 0 );
  unpackedSize += edgeManager->Unpack( receiveBufferPtr, newGhostEdges, 0 );
  unpackedSize += faceManager->Unpack( receiveBufferPtr, newGhostFaces, 0 );
  unpackedSize += elemManager->Unpack( receiveBufferPtr, newGhostElems );

  unpackedSize += nodeManager->UnpackUpDownMaps( receiveBufferPtr, modGhostNodes, false, true );
  unpackedSize += edgeManager->UnpackUpDownMaps( receiveBufferPtr, modGhostEdges, false, true );
  unpackedSize += faceManager->UnpackUpDownMaps( receiveBufferPtr, modGhostFaces, false, true );
  unpackedSize += elemManager->UnpackUpDownMaps( receiveBufferPtr, modGhostElems, true );

  unpackedSize += nodeManager->UnpackParentChildMaps( receiveBufferPtr, modGhostNodes );
  unpackedSize += edgeManager->UnpackParentChildMaps( receiveBufferPtr, modGhostEdges );
  unpackedSize += faceManager->UnpackParentChildMaps( receiveBufferPtr, modGhostFaces );


  for( localIndex a=0 ; a<newGhostNodes.size() ; ++a )
  {
    nodeGhostsToRecv.push_back(newGhostNodes[a]);
  }

  for( localIndex a=0 ; a<newGhostEdges.size() ; ++a )
  {
    edgeGhostsToRecv.push_back(newGhostEdges[a]);
  }

  for( localIndex a=0 ; a<newGhostFaces.size() ; ++a )
  {
    faceGhostsToRecv.push_back(newGhostFaces[a]);
  }

  for( localIndex er=0 ; er<elemManager->numRegions() ; ++er )
  {
    ElementRegionBase * const elemRegion = elemManager->GetRegion(er);
    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      for( localIndex const & newElemIndex : newGhostElemsData[er][esr] )
      {
        elementGhostToReceive[er][esr].get().push_back( newElemIndex);
        receivedObjects.newElements[{er,esr}].insert(newElemIndex);
      }
      receivedObjects.modifiedElements[{er,esr}].insert( modGhostElemsData[er][esr].begin(),
                                                         modGhostElemsData[er][esr].end() );
    }
  }


  receivedObjects.newNodes.insert( newGhostNodes.begin(), newGhostNodes.end() );
  receivedObjects.modifiedNodes.insert( modGhostNodes.begin(), modGhostNodes.end() );
  receivedObjects.newEdges.insert( newGhostEdges.begin(), newGhostEdges.end() );
  receivedObjects.modifiedEdges.insert( modGhostEdges.begin(), modGhostEdges.end() );
  receivedObjects.newFaces.insert( newGhostFaces.begin(), newGhostFaces.end() );
  receivedObjects.modifiedFaces.insert( modGhostFaces.begin(), modGhostFaces.end() );

}

void ParallelTopologyChange::updateConnectorsToFaceElems( std::set<localIndex> const & newFaceElements,
                                                          FaceElementSubRegion const * const faceElemSubRegion,
                                                          EdgeManager * const edgeManager )
{
  ArrayOfArrays<localIndex> & connectorToElem = edgeManager->m_fractureConnectorEdgesToFaceElements;
  map< localIndex, localIndex > & edgesToConnectorEdges = edgeManager->m_edgesToFractureConnectorsEdges;
  array1d<localIndex> & connectorEdgesToEdges = edgeManager->m_fractureConnectorsEdgesToEdges;

  arrayView1d< arrayView1d< localIndex const > const > const & facesToEdges = faceElemSubRegion->edgeList().toViewConst();

  for( localIndex const & kfe : newFaceElements )
  {
    arrayView1d<localIndex const> const & faceToEdges = facesToEdges[kfe];
    for( localIndex ke=0 ; ke<faceToEdges.size() ; ++ke )
    {
      localIndex const edgeIndex = faceToEdges[ke];

      auto connIter = edgesToConnectorEdges.find(edgeIndex);
      if( connIter==edgesToConnectorEdges.end() )
      {
        connectorToElem.resize( connectorToElem.size() + 1 );
        connectorEdgesToEdges.push_back(edgeIndex);
        edgesToConnectorEdges[edgeIndex] = connectorEdgesToEdges.size() - 1;
      }
      localIndex const connectorIndex = edgesToConnectorEdges.at(edgeIndex);

      localIndex const numExistingCells = connectorToElem.sizeOfArray(connectorIndex);
      bool cellExistsInMap = false;
      for( localIndex k=0 ; k<numExistingCells ; ++k )
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
        edgeManager-> m_recalculateFractureConnectorEdges.insert( connectorIndex );
      }
    }
  }
}

} /* namespace geosx */
