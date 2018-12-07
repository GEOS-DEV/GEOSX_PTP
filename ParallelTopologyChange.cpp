/*
 * ParallelTopologyChange.cpp
 *
 *  Created on: Sep 20, 2018
 *      Author: settgast1
 */

#include "ParallelTopologyChange.hpp"

#include "common/TimingMacros.hpp"
#include "MPI_Communications/CommunicationTools.hpp"
#include "mesh/ElementRegionManager.hpp"


namespace geosx
{

using namespace dataRepository;

ParallelTopologyChange::ParallelTopologyChange()
{
  // TODO Auto-generated constructor stub

}

ParallelTopologyChange::~ParallelTopologyChange()
{
  // TODO Auto-generated destructor stub
}

void ParallelTopologyChange::SyncronizeTopologyChange( MeshLevel * const mesh,
                                                       array1d<NeighborCommunicator> & neighbors,
                                                       ModifiedObjectLists const & modifiedObjects )
{

  NodeManager * const nodeManager = mesh->getNodeManager();
  EdgeManager * const edgeManager = mesh->getEdgeManager();
  FaceManager * const faceManager = mesh->getFaceManager();
  ElementRegionManager * const elemManager = mesh->getElemManager();

  array1d<integer> & nodeGhostRank = nodeManager->GhostRank();
  array1d<integer> & edgeGhostRank = edgeManager->GhostRank();
  array1d<integer> & faceGhostRank = faceManager->GhostRank();

  //************************************************************************************************
  // 1) Assign new global indices to the new objects
  CommunicationTools::AssignNewGlobalIndices( *nodeManager, modifiedObjects.newNodes );
  CommunicationTools::AssignNewGlobalIndices( *edgeManager, modifiedObjects.newEdges );
  CommunicationTools::AssignNewGlobalIndices( *faceManager, modifiedObjects.newFaces );



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
    NeighborCommunicator& neighbor = neighbors[neighborIndex];
    int const neighborRank = neighbor.NeighborRank();

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


  set<localIndex> allNewNodes, allModifiedNodes;
  set<localIndex> allNewEdges, allModifiedEdges;
  set<localIndex> allNewFaces, allModifiedFaces;


  allNewNodes.insert( modifiedObjects.newNodes.begin(),
                      modifiedObjects.newNodes.end() );
  allModifiedNodes.insert( modifiedObjects.modifiedNodes.begin(),
                           modifiedObjects.modifiedNodes.end() );

  allNewEdges.insert( modifiedObjects.newEdges.begin(),
                      modifiedObjects.newEdges.end() );
  allModifiedEdges.insert( modifiedObjects.modifiedEdges.begin(),
                           modifiedObjects.modifiedEdges.end() );

  allNewFaces.insert( modifiedObjects.newFaces.begin(),
                      modifiedObjects.newFaces.end() );
  allModifiedFaces.insert( modifiedObjects.modifiedFaces.begin(),
                           modifiedObjects.modifiedFaces.end() );

  ElementRegionManager::ElementReferenceAccessor< localIndex_array > modifiedElements;
  array1d<array1d<localIndex_array> > modifiedElementsData;
  array1d<array1d< set<localIndex> > > allModifiedElements;

  modifiedElements.resize(elemManager->numRegions());
  modifiedElementsData.resize(elemManager->numRegions());
  allModifiedElements.resize(elemManager->numRegions());
  for( localIndex er=0 ; er<elemManager->numRegions() ; ++er )
  {
    ElementRegion * const elemRegion = elemManager->GetRegion(er);
    modifiedElements[er].resize(elemRegion->numSubRegions());
    modifiedElementsData[er].resize(elemRegion->numSubRegions());
    allModifiedElements[er].resize(elemRegion->numSubRegions());
    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      CellBlockSubRegion * const subRegion = elemRegion->GetSubRegion(esr);

      modifiedElements[er][esr].set( modifiedElementsData[er][esr] );

      std::map< std::string, set<localIndex> >::const_iterator
      iter = modifiedObjects.modifiedElements.find(subRegion->getName());
      if( iter != modifiedObjects.modifiedElements.end() )
      {
      allModifiedElements[er][esr].insert( iter->second.begin(),
                                           iter->second.end() );
      }

    }
  }


  // unpack the buffers and get lists of the new objects.
  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
  {
    localIndex_array newLocalNodes, modifiedLocalNodes;

    localIndex_array newLocalEdges, modifiedLocalEdges;

    localIndex_array newLocalFaces, modifiedLocalFaces;


    int neighborIndex;
    MPI_Waitany( commData.size,
                 commData.mpiRecvBufferRequest.data(),
                 &neighborIndex,
                 commData.mpiRecvBufferStatus.data() );

    NeighborCommunicator& neighbor = neighbors[neighborIndex];

    buffer_type const & receiveBuffer = neighbor.ReceiveBuffer( commData.commID );
    buffer_unit_type const * receiveBufferPtr = receiveBuffer.data();

    int unpackedSize = 0;
    unpackedSize += nodeManager->UnpackGlobalMaps( receiveBufferPtr, newLocalNodes, 0 );
    unpackedSize += edgeManager->UnpackGlobalMaps( receiveBufferPtr, newLocalEdges, 0 );
    unpackedSize += faceManager->UnpackGlobalMaps( receiveBufferPtr, newLocalFaces, 0 );

    unpackedSize += nodeManager->UnpackUpDownMaps( receiveBufferPtr, newLocalNodes );
    unpackedSize += edgeManager->UnpackUpDownMaps( receiveBufferPtr, newLocalEdges );
    unpackedSize += faceManager->UnpackUpDownMaps( receiveBufferPtr, newLocalFaces );

    unpackedSize += nodeManager->Unpack( receiveBufferPtr, newLocalNodes, 0 );
    unpackedSize += edgeManager->Unpack( receiveBufferPtr, newLocalEdges, 0 );
    unpackedSize += faceManager->Unpack( receiveBufferPtr, newLocalFaces, 0 );

    unpackedSize += nodeManager->UnpackUpDownMaps( receiveBufferPtr, modifiedLocalNodes );
    unpackedSize += edgeManager->UnpackUpDownMaps( receiveBufferPtr, modifiedLocalEdges );
    unpackedSize += faceManager->UnpackUpDownMaps( receiveBufferPtr, modifiedLocalFaces );
    unpackedSize += elemManager->UnpackUpDownMaps( receiveBufferPtr, modifiedElements );

    unpackedSize += nodeManager->Unpack( receiveBufferPtr, modifiedLocalNodes, 0 );
    unpackedSize += edgeManager->Unpack( receiveBufferPtr, modifiedLocalEdges, 0 );
    unpackedSize += faceManager->Unpack( receiveBufferPtr, modifiedLocalFaces, 0 );
//    unpackedSize += elemManager->Unpack( receiveBufferPtr, modifiedElements );

    allNewNodes.insert( newLocalNodes.begin(), newLocalNodes.end() );
    allModifiedNodes.insert( modifiedLocalNodes.begin(), modifiedLocalNodes.end() );

    allNewEdges.insert( newLocalEdges.begin(), newLocalEdges.end() );
    allModifiedEdges.insert( modifiedLocalEdges.begin(), modifiedLocalEdges.end() );

    allNewFaces.insert( newLocalFaces.begin(), newLocalFaces.end() );
    allModifiedFaces.insert( modifiedLocalFaces.begin(), modifiedLocalFaces.end() );

    for( localIndex er=0 ; er<elemManager->numRegions() ; ++er )
    {
      ElementRegion * const elemRegion = elemManager->GetRegion(er);
      for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
      {
        allModifiedElements[er][esr].insert( modifiedElements[er][esr].get().begin(),
                                             modifiedElements[er][esr].get().end() );
      }
    }
  }

  MPI_Waitall( commData.size,
               commData.mpiSizeSendBufferRequest.data(),
               commData.mpiSizeSendBufferStatus.data() );

  MPI_Waitall( commData.size,
               commData.mpiSendBufferRequest.data(),
               commData.mpiSizeSendBufferStatus.data() );



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
                        allNewNodes,
                        allNewEdges,
                        allNewFaces,
                        allModifiedNodes,
                        allModifiedEdges,
                        allModifiedFaces,
                        allModifiedElements );

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
    UnpackNewModToGhosts( &neighbor, commData2.commID, mesh );
  }

  nodeManager->FixUpDownMaps(false);
  edgeManager->FixUpDownMaps(false);
  faceManager->FixUpDownMaps(false);
  for( localIndex er=0 ; er<elemManager->numRegions() ; ++er )
  {
    ElementRegion * const elemRegion = elemManager->GetRegion(er);
    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      CellBlockSubRegion * const subRegion = elemRegion->GetSubRegion(esr);
      subRegion->FixUpDownMaps(false);
    }
  }

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

  int const neighborRank = neighbor->NeighborRank();

  array1d<localIndex> newNodePackListArray(modifiedObjects.newNodes.size());
  {
    localIndex a=0 ;
    for( auto const index : modifiedObjects.newNodes )
    {
      if( nodeGhostRank[index] == neighborRank )
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
      if( nodeGhostRank[index] == neighborRank )
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
      if( edgeGhostRank[index] == neighborRank )
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
      if( edgeGhostRank[index] == neighborRank )
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
      if( faceGhostRank[index] == neighborRank )
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
      if( faceGhostRank[index] == neighborRank )
      {
        modFacePackListArray[a] = index;
        ++a;
      }
    }
    modFacePackListArray.resize(a);
  }


  ElementRegionManager::ElementReferenceAccessor< localIndex_array > modElemPackList;
  array1d<array1d<localIndex_array> > modElemData;
  modElemPackList.resize(elemManager.numRegions());
  modElemData.resize(elemManager.numRegions());
  for( localIndex er=0 ; er<elemManager.numRegions() ; ++er )
  {
    ElementRegion * const elemRegion = elemManager.GetRegion(er);
    modElemPackList[er].resize(elemRegion->numSubRegions());
    modElemData[er].resize(elemRegion->numSubRegions());
    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      CellBlockSubRegion * const subRegion = elemRegion->GetSubRegion(esr);
      integer_array const & subRegionGhostRank = subRegion->GhostRank();
      string const & subRegionName = subRegion->getName();
      if( modifiedObjects.modifiedElements.count(subRegionName) > 0 )
      {
        set<localIndex> const & elemList = modifiedObjects.modifiedElements.at(subRegionName);

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

      modElemPackList[er][esr].set( modElemData[er][esr] );
    }
  }








  bufferSize += nodeManager.PackGlobalMapsSize( newNodePackListArray, 0 );
  bufferSize += edgeManager.PackGlobalMapsSize( newEdgePackListArray, 0 );
  bufferSize += faceManager.PackGlobalMapsSize( newFacePackListArray, 0 );

  bufferSize += nodeManager.PackUpDownMapsSize( newNodePackListArray );
  bufferSize += edgeManager.PackUpDownMapsSize( newEdgePackListArray );
  bufferSize += faceManager.PackUpDownMapsSize( newFacePackListArray );

  bufferSize += nodeManager.PackSize( {}, newNodePackListArray, 0 );
  bufferSize += edgeManager.PackSize( {}, newEdgePackListArray, 0 );
  bufferSize += faceManager.PackSize( {}, newFacePackListArray, 0 );

  bufferSize += nodeManager.PackUpDownMapsSize( modNodePackListArray );
  bufferSize += edgeManager.PackUpDownMapsSize( modEdgePackListArray );
  bufferSize += faceManager.PackUpDownMapsSize( modFacePackListArray );
  bufferSize += elemManager.PackUpDownMapsSize( modElemPackList );

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

  packedSize += nodeManager.PackUpDownMaps( sendBufferPtr, newNodePackListArray );
  packedSize += edgeManager.PackUpDownMaps( sendBufferPtr, newEdgePackListArray );
  packedSize += faceManager.PackUpDownMaps( sendBufferPtr, newFacePackListArray );

  packedSize += nodeManager.Pack( sendBufferPtr, {}, newNodePackListArray, 0 );
  packedSize += edgeManager.Pack( sendBufferPtr, {}, newEdgePackListArray, 0 );
  packedSize += faceManager.Pack( sendBufferPtr, {}, newFacePackListArray, 0 );

  packedSize += nodeManager.PackUpDownMaps( sendBufferPtr, modNodePackListArray );
  packedSize += edgeManager.PackUpDownMaps( sendBufferPtr, modEdgePackListArray );
  packedSize += faceManager.PackUpDownMaps( sendBufferPtr, modFacePackListArray );
  packedSize += elemManager.PackUpDownMaps( sendBufferPtr, modElemPackList );

  packedSize += nodeManager.Pack( sendBufferPtr, {}, modNodePackListArray, 0 );
  packedSize += edgeManager.Pack( sendBufferPtr, {}, modEdgePackListArray, 0 );
  packedSize += faceManager.Pack( sendBufferPtr, {}, modFacePackListArray, 0 );


  GEOS_ERROR_IF( bufferSize != packedSize, "Allocated Buffer Size is not equal to packed buffer size" );


}

void ParallelTopologyChange::UnpackNewAndModifiedObjects( NeighborCommunicator * const neighbor,
                                                          MeshLevel * const mesh,
                                                          int const commID )
{
  GEOSX_MARK_FUNCTION;
  NodeManager & nodeManager = *(mesh->getNodeManager());
  EdgeManager & edgeManager = *(mesh->getEdgeManager());
  FaceManager & faceManager = *(mesh->getFaceManager());
  ElementRegionManager & elemManager = *(mesh->getElemManager());

  buffer_type const & receiveBuffer = neighbor->ReceiveBuffer( commID );
  buffer_unit_type const * receiveBufferPtr = receiveBuffer.data();


  int unpackedSize = 0;

  localIndex_array nodeUnpackList;
  unpackedSize += nodeManager.UnpackGlobalMaps( receiveBufferPtr, nodeUnpackList, 0 );

  localIndex_array edgeUnpackList;
  unpackedSize += edgeManager.UnpackGlobalMaps( receiveBufferPtr, edgeUnpackList, 0 );

  localIndex_array faceUnpackList;
  unpackedSize += faceManager.UnpackGlobalMaps( receiveBufferPtr, faceUnpackList, 0 );

  ElementRegionManager::ElementReferenceAccessor<localIndex_array>
  elementAdjacencyReceiveList =
    elemManager.ConstructReferenceAccessor<localIndex_array>( ObjectManagerBase::viewKeyStruct::ghostsToReceiveString,
                                                         std::to_string( neighbor->NeighborRank() ) );
  unpackedSize += elemManager.UnpackGlobalMaps( receiveBufferPtr, elementAdjacencyReceiveList );


  unpackedSize += nodeManager.UnpackUpDownMaps( receiveBufferPtr, nodeUnpackList );
  unpackedSize += edgeManager.UnpackUpDownMaps( receiveBufferPtr, edgeUnpackList );
  unpackedSize += faceManager.UnpackUpDownMaps( receiveBufferPtr, faceUnpackList );
  unpackedSize += elemManager.UnpackUpDownMaps( receiveBufferPtr, elementAdjacencyReceiveList );


  unpackedSize += nodeManager.Unpack( receiveBufferPtr, nodeUnpackList, 0 );
  unpackedSize += edgeManager.Unpack( receiveBufferPtr, edgeUnpackList, 0 );
  unpackedSize += faceManager.Unpack( receiveBufferPtr, faceUnpackList, 0 );
  unpackedSize += elemManager.Unpack( receiveBufferPtr, elementAdjacencyReceiveList );

}


static void FilterNewObjectsForPackToGhosts( set<localIndex> const & objectList,
                                             localIndex_array const & parentIndices,
                                             localIndex_array & ghostsToSend,
                                             localIndex_array & objectsToSend )
{
  for( auto const index : objectList )
  {
    localIndex const parentIndex = parentIndices[index];
    for( localIndex a=0 ; a<ghostsToSend.size() ; ++a )
    {
      if( ghostsToSend[a]==parentIndex )
      {
        objectsToSend.push_back( index );
        break;
      }
    }
  }
  for( localIndex a=0 ; a<objectsToSend.size() ; ++a )
  {
    ghostsToSend.push_back( objectsToSend[a] );
  }
}

static void FilterModObjectsForPackToGhosts( set<localIndex> const & objectList,
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
                                                             set<localIndex> const & allNewNodes,
                                                             set<localIndex> const & allNewEdges,
                                                             set<localIndex> const & allNewFaces,
                                                             set<localIndex> const & allModNodes,
                                                             set<localIndex> const & allModEdges,
                                                             set<localIndex> const & allModFaces,
                                                             array1d<array1d< set<localIndex> > > const & allModElems )
{
  NodeManager * const nodeManager = mesh->getNodeManager();
  EdgeManager * const edgeManager = mesh->getEdgeManager();
  FaceManager * const faceManager = mesh->getFaceManager();
  ElementRegionManager * const elemManager = mesh->getElemManager();

  array1d<integer> & nodeGhostRank = nodeManager->GhostRank();
  array1d<integer> & edgeGhostRank = edgeManager->GhostRank();
  array1d<integer> & faceGhostRank = faceManager->GhostRank();

  localIndex_array newNodesToSend;
  localIndex_array newEdgesToSend;
  localIndex_array newFacesToSend;

  localIndex_array modNodesToSend;
  localIndex_array modEdgesToSend;
  localIndex_array modFacesToSend;
  ElementRegionManager::ElementReferenceAccessor<localIndex_array> modElemsToSend;
  array1d< array1d <localIndex_array> > elemsToSendData;

  ManagedGroup *
   nodeNeighborData = nodeManager->GetGroup( nodeManager->groupKeys.neighborData )->
                      GetGroup( std::to_string( neighbor->NeighborRank() ) );

   ManagedGroup *
   edgeNeighborData = edgeManager->GetGroup( edgeManager->groupKeys.neighborData )->
                      GetGroup( std::to_string( neighbor->NeighborRank() ) );

   ManagedGroup *
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


   FilterNewObjectsForPackToGhosts( allNewNodes, nodalParentIndices, nodeGhostsToSend, newNodesToSend );
   FilterModObjectsForPackToGhosts( allModNodes, nodeGhostsToSend, modNodesToSend );

   FilterNewObjectsForPackToGhosts( allNewEdges, edgeParentIndices, edgeGhostsToSend, newEdgesToSend );
   FilterModObjectsForPackToGhosts( allModEdges, edgeGhostsToSend, modEdgesToSend );

   FilterNewObjectsForPackToGhosts( allNewFaces, faceParentIndices, faceGhostsToSend, newFacesToSend );
   FilterModObjectsForPackToGhosts( allModFaces, faceGhostsToSend, modFacesToSend );


   elemsToSendData.resize( elemManager->numRegions() );
   modElemsToSend.resize( elemManager->numRegions() );
   for( localIndex er=0 ; er<elemManager->numRegions() ; ++er )
   {
     ElementRegion * const elemRegion = elemManager->GetRegion(er);
     elemsToSendData[er].resize( elemRegion->numSubRegions() );
     modElemsToSend[er].resize( elemRegion->numSubRegions() );
     for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
     {
       modElemsToSend[er][esr].set( elemsToSendData[er][esr] );
       for( localIndex a=0 ; a<elementGhostToSend[er][esr].get().size() ; ++a )
       {
         if( allModElems[er][esr].count( elementGhostToSend[er][esr][a] ) > 0 )
         {
           elemsToSendData[er][esr].push_back(elementGhostToSend[er][esr][a]);
         }
       }
     }
   }


   int bufferSize = 0;

   bufferSize += nodeManager->PackGlobalMapsSize( newNodesToSend, 0 );
   bufferSize += edgeManager->PackGlobalMapsSize( newEdgesToSend, 0 );
   bufferSize += faceManager->PackGlobalMapsSize( newFacesToSend, 0 );

   bufferSize += nodeManager->PackUpDownMapsSize( newNodesToSend );
   bufferSize += edgeManager->PackUpDownMapsSize( newEdgesToSend );
   bufferSize += faceManager->PackUpDownMapsSize( newFacesToSend );

   bufferSize += nodeManager->PackSize( {}, newNodesToSend, 0 );
   bufferSize += edgeManager->PackSize( {}, newEdgesToSend, 0 );
   bufferSize += faceManager->PackSize( {}, newFacesToSend, 0 );

   bufferSize += nodeManager->PackUpDownMapsSize( modNodesToSend );
   bufferSize += edgeManager->PackUpDownMapsSize( modEdgesToSend );
   bufferSize += faceManager->PackUpDownMapsSize( modFacesToSend );
   bufferSize += elemManager->PackUpDownMapsSize( modElemsToSend );



   neighbor->resizeSendBuffer( commID, bufferSize );

   buffer_type & sendBuffer = neighbor->SendBuffer( commID );
   buffer_unit_type * sendBufferPtr = sendBuffer.data();

   int packedSize = 0;

   packedSize += nodeManager->PackGlobalMaps( sendBufferPtr, newNodesToSend, 0 );
   packedSize += edgeManager->PackGlobalMaps( sendBufferPtr, newEdgesToSend, 0 );
   packedSize += faceManager->PackGlobalMaps( sendBufferPtr, newFacesToSend, 0 );

   packedSize += nodeManager->PackUpDownMaps( sendBufferPtr, newNodesToSend );
   packedSize += edgeManager->PackUpDownMaps( sendBufferPtr, newEdgesToSend );
   packedSize += faceManager->PackUpDownMaps( sendBufferPtr, newFacesToSend );

   packedSize += nodeManager->Pack( sendBufferPtr, {}, newNodesToSend, 0 );
   packedSize += edgeManager->Pack( sendBufferPtr, {}, newEdgesToSend, 0 );
   packedSize += faceManager->Pack( sendBufferPtr, {}, newFacesToSend, 0 );

   packedSize += nodeManager->PackUpDownMaps( sendBufferPtr, modNodesToSend );
   packedSize += edgeManager->PackUpDownMaps( sendBufferPtr, modEdgesToSend );
   packedSize += faceManager->PackUpDownMaps( sendBufferPtr, modFacesToSend );
   packedSize += elemManager->PackUpDownMaps( sendBufferPtr, modElemsToSend );


   GEOS_ERROR_IF( bufferSize != packedSize, "Allocated Buffer Size is not equal to packed buffer size" );

   neighbor->MPI_iSendReceive( commID, MPI_COMM_GEOSX );


}

void ParallelTopologyChange::UnpackNewModToGhosts( NeighborCommunicator * const neighbor,
                                                   int commID,
                                                   MeshLevel * const mesh )
{
  int unpackedSize = 0;

  NodeManager * const nodeManager = mesh->getNodeManager();
  EdgeManager * const edgeManager = mesh->getEdgeManager();
  FaceManager * const faceManager = mesh->getFaceManager();
  ElementRegionManager * const elemManager = mesh->getElemManager();


  ManagedGroup *
   nodeNeighborData = nodeManager->GetGroup( nodeManager->groupKeys.neighborData )->
                      GetGroup( std::to_string( neighbor->NeighborRank() ) );

   ManagedGroup *
   edgeNeighborData = edgeManager->GetGroup( edgeManager->groupKeys.neighborData )->
                      GetGroup( std::to_string( neighbor->NeighborRank() ) );

   ManagedGroup *
   faceNeighborData = faceManager->GetGroup( faceManager->groupKeys.neighborData )->
                      GetGroup( std::to_string( neighbor->NeighborRank() ) );

   localIndex_array &
   nodeGhostsToRecv = nodeNeighborData->getReference<localIndex_array>( nodeManager->viewKeys.ghostsToReceive );

   localIndex_array &
   edgeGhostsToRecv = edgeNeighborData->getReference<localIndex_array>( nodeManager->viewKeys.ghostsToReceive );

   localIndex_array &
   faceGhostsToRecv = faceNeighborData->getReference<localIndex_array>( faceManager->viewKeys.ghostsToReceive );





  buffer_type const & receiveBuffer = neighbor->ReceiveBuffer( commID );
  buffer_unit_type const * receiveBufferPtr = receiveBuffer.data();

  localIndex_array newGhostNodes;
  localIndex_array newGhostEdges;
  localIndex_array newGhostFaces;

  localIndex_array modGhostNodes;
  localIndex_array modGhostEdges;
  localIndex_array modGhostFaces;

  ElementRegionManager::ElementReferenceAccessor<localIndex_array> modGhostElems;
  array1d< array1d <localIndex_array> > modGhostElemsData;
  modGhostElems.resize( elemManager->numRegions() );
  modGhostElemsData.resize( elemManager->numRegions() );
  for( localIndex er=0 ; er<elemManager->numRegions() ; ++er )
  {
    ElementRegion * const elemRegion = elemManager->GetRegion(er);
    modGhostElemsData[er].resize( elemRegion->numSubRegions() );
    modGhostElems[er].resize( elemRegion->numSubRegions() );
    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      modGhostElems[er][esr].set( modGhostElemsData[er][esr] );
    }
  }


  unpackedSize += nodeManager->UnpackGlobalMaps( receiveBufferPtr, newGhostNodes, 0 );
  unpackedSize += edgeManager->UnpackGlobalMaps( receiveBufferPtr, newGhostEdges, 0 );
  unpackedSize += faceManager->UnpackGlobalMaps( receiveBufferPtr, newGhostFaces, 0 );

  unpackedSize += nodeManager->UnpackUpDownMaps( receiveBufferPtr, newGhostNodes );
  unpackedSize += edgeManager->UnpackUpDownMaps( receiveBufferPtr, newGhostEdges );
  unpackedSize += faceManager->UnpackUpDownMaps( receiveBufferPtr, newGhostFaces );

  unpackedSize += nodeManager->Unpack( receiveBufferPtr, newGhostNodes, 0 );
  unpackedSize += edgeManager->Unpack( receiveBufferPtr, newGhostEdges, 0 );
  unpackedSize += faceManager->Unpack( receiveBufferPtr, newGhostFaces, 0 );

  unpackedSize += nodeManager->UnpackUpDownMaps( receiveBufferPtr, modGhostNodes );
  unpackedSize += edgeManager->UnpackUpDownMaps( receiveBufferPtr, modGhostEdges );
  unpackedSize += faceManager->UnpackUpDownMaps( receiveBufferPtr, modGhostFaces );
  unpackedSize += elemManager->UnpackUpDownMaps( receiveBufferPtr, modGhostElems );


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
}


} /* namespace geosx */
