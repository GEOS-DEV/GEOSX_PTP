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
  for( unsigned int neighborIndex=0 ; neighborIndex<neighbors.size() ; ++neighborIndex )
  {
    NeighborCommunicator& neighbor = neighbors[neighborIndex];
    int const neighborRank = neighbor.NeighborRank();



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
    }


    ElementRegionManager::ElementViewAccessor< localIndex_array > modElemPackList;
    array1d<array1d<localIndex_array> > modElemData;
    modElemData.resize(elemManager->numRegions());
    for( localIndex er=0 ; er<elemManager->numRegions() ; ++er )
    {
      ElementRegion * const elemRegion = elemManager->GetRegion(er);
      modElemData[er].resize(elemRegion->numSubRegions());
      for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
      {
        CellBlockSubRegion * const subRegion = elemRegion->GetSubRegion(esr);
        integer_array const & subRegionGhostRank = subRegion->GhostRank();
        string const & subRegionName = subRegion->getName();
        set<localIndex> const & elemList = modifiedObjects.modifiedElements.at(subRegionName);

        modElemData[er][esr].resize( elemList.size() );

        localIndex a;
        for( auto const index : elemList )
        {
          if( subRegionGhostRank[index] == neighborRank )
          {
            modElemData[er][esr][a] = index;
            ++a;
          }
        }

        modElemPackList[er][esr].set( modElemData[er][esr] );
      }
    }


    PackNewAndModifiedObjects( &neighbor,mesh,
                                        newNodePackListArray,
                                        newEdgePackListArray,
                                        newFacePackListArray,
                                        modNodePackListArray,
                                        modEdgePackListArray,
                                        modFacePackListArray,
                                        modElemPackList,
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

  ElementRegionManager::ElementViewAccessor< localIndex_array > modifiedElements;
  array1d<array1d<localIndex_array> > modifiedElementsData;
  array1d<array1d< set<localIndex> > > allModifiedElements;

  modifiedElements.resize(elemManager->numRegions());
  allModifiedElements.resize(elemManager->numRegions());
  for( localIndex er=0 ; er<elemManager->numRegions() ; ++er )
  {
    ElementRegion * const elemRegion = elemManager->GetRegion(er);
    modifiedElements[er].resize(elemRegion->numSubRegions());
    allModifiedElements[er].resize(elemRegion->numSubRegions());
    for( localIndex esr=0 ; esr<elemRegion->numSubRegions() ; ++esr )
    {
      CellBlockSubRegion * const subRegion = elemRegion->GetSubRegion(esr);

      modifiedElements[er][esr].set( modifiedElementsData[er][esr] );
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
    unpackedSize += elemManager->Unpack( receiveBufferPtr, modifiedElements );

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
  for( unsigned int neighborIndex=0 ; neighborIndex<neighbors.size() ; ++neighborIndex )
  {
    NeighborCommunicator& neighbor = neighbors[neighborIndex];

    PackNewModToGhosts( &neighbor,
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


}


void ParallelTopologyChange::PackNewAndModifiedObjects(
    NeighborCommunicator * const neighbor,
    MeshLevel * const meshLevel,
    array1d<localIndex> const & newNodePackList,
    array1d<localIndex> const & newEdgePackList,
    array1d<localIndex> const & newFacePackList,
    array1d<localIndex> const & modNodePackList,
    array1d<localIndex> const & modEdgePackList,
    array1d<localIndex> const & modFacePackList,
    array1d< array1d< ReferenceWrapper< localIndex_array > > > const & modifiedElements,
    int const commID )
{
  int bufferSize = 0;

  NodeManager & nodeManager = *(meshLevel->getNodeManager());
  EdgeManager & edgeManager = *(meshLevel->getEdgeManager());
  FaceManager & faceManager = *(meshLevel->getFaceManager());
  ElementRegionManager & elemManager = *(meshLevel->getElemManager() );


  bufferSize += nodeManager.PackGlobalMapsSize( newNodePackList, 0 );
  bufferSize += edgeManager.PackGlobalMapsSize( newEdgePackList, 0 );
  bufferSize += faceManager.PackGlobalMapsSize( newFacePackList, 0 );

  bufferSize += nodeManager.PackUpDownMapsSize( newNodePackList );
  bufferSize += edgeManager.PackUpDownMapsSize( newEdgePackList );
  bufferSize += faceManager.PackUpDownMapsSize( newFacePackList );

  bufferSize += nodeManager.PackSize( {}, newNodePackList, 0 );
  bufferSize += edgeManager.PackSize( {}, newEdgePackList, 0 );
  bufferSize += faceManager.PackSize( {}, newFacePackList, 0 );

  bufferSize += nodeManager.PackUpDownMapsSize( modNodePackList );
  bufferSize += edgeManager.PackUpDownMapsSize( modEdgePackList );
  bufferSize += faceManager.PackUpDownMapsSize( modFacePackList );
  bufferSize += elemManager.PackUpDownMapsSize( modifiedElements );

  bufferSize += nodeManager.PackSize( {}, modNodePackList, 0 );
  bufferSize += edgeManager.PackSize( {}, modEdgePackList, 0 );
  bufferSize += faceManager.PackSize( {}, modFacePackList, 0 );



  neighbor->resizeSendBuffer( commID, bufferSize );

  buffer_type & sendBuffer = neighbor->SendBuffer( commID );
  buffer_unit_type * sendBufferPtr = sendBuffer.data();

  int packedSize = 0;

  packedSize += nodeManager.PackGlobalMaps( sendBufferPtr, newNodePackList, 0 );
  packedSize += edgeManager.PackGlobalMaps( sendBufferPtr, newEdgePackList, 0 );
  packedSize += faceManager.PackGlobalMaps( sendBufferPtr, newFacePackList, 0 );

  packedSize += nodeManager.PackUpDownMaps( sendBufferPtr, newNodePackList );
  packedSize += edgeManager.PackUpDownMaps( sendBufferPtr, newEdgePackList );
  packedSize += faceManager.PackUpDownMaps( sendBufferPtr, newFacePackList );

  packedSize += nodeManager.Pack( sendBufferPtr, {}, newNodePackList, 0 );
  packedSize += edgeManager.Pack( sendBufferPtr, {}, newEdgePackList, 0 );
  packedSize += faceManager.Pack( sendBufferPtr, {}, newFacePackList, 0 );

  packedSize += nodeManager.PackUpDownMaps( sendBufferPtr, modNodePackList );
  packedSize += edgeManager.PackUpDownMaps( sendBufferPtr, modEdgePackList );
  packedSize += faceManager.PackUpDownMaps( sendBufferPtr, modFacePackList );
  packedSize += elemManager.PackUpDownMaps( sendBufferPtr, modifiedElements );

  packedSize += nodeManager.Pack( sendBufferPtr, {}, modNodePackList, 0 );
  packedSize += edgeManager.Pack( sendBufferPtr, {}, modEdgePackList, 0 );
  packedSize += faceManager.Pack( sendBufferPtr, {}, modFacePackList, 0 );


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

  ElementRegionManager::ElementViewAccessor<localIndex_array>
  elementAdjacencyReceiveList =
    elemManager.ConstructViewAccessor<localIndex_array>( ObjectManagerBase::viewKeyStruct::ghostsToReceiveString,
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


void ParallelTopologyChange::PackNewModToGhosts( NeighborCommunicator * const neighbor,
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
  ElementRegionManager::ElementViewAccessor<localIndex_array> modElemsToSend;
  array1d< array1d <localIndex_array> > elemsToSendData;

  ManagedGroup * const
   nodeNeighborData = nodeManager->GetGroup( nodeManager->groupKeys.neighborData )->
                      GetGroup( std::to_string( neighbor->NeighborRank() ) );

   ManagedGroup * const
   edgeNeighborData = edgeManager->GetGroup( edgeManager->groupKeys.neighborData )->
                      GetGroup( std::to_string( neighbor->NeighborRank() ) );

   ManagedGroup * const
   faceNeighborData = faceManager->GetGroup( faceManager->groupKeys.neighborData )->
                      GetGroup( std::to_string( neighbor->NeighborRank() ) );

   localIndex_array const &
   nodeGhostsToSend = nodeNeighborData->getReference<localIndex_array>( nodeManager->viewKeys.ghostsToSend );

   localIndex_array const &
   edgeGhostsToSend = edgeNeighborData->getReference<localIndex_array>( nodeManager->viewKeys.ghostsToSend );

   localIndex_array const &
   faceGhostsToSend = faceNeighborData->getReference<localIndex_array>( faceManager->viewKeys.ghostsToSend );

   ElementRegionManager::ElementViewAccessor<localIndex_array>
   elementGhostToSend =
     elemManager->ConstructViewAccessor<localIndex_array>( ObjectManagerBase::
                                                          viewKeyStruct::
                                                          ghostsToSendString,
                                                          std::to_string( neighbor->NeighborRank() ) );



   for( localIndex a=0 ; a<nodeGhostsToSend.size() ; ++a )
   {
     if( allNewNodes.count(nodeGhostsToSend[a]) > 0)
     {
       newNodesToSend.push_back( nodeGhostsToSend[a] );
     }
     if( allModNodes.count(nodeGhostsToSend[a]) > 0)
     {
       modNodesToSend.push_back( nodeGhostsToSend[a] );
     }
   }

   for( localIndex a=0 ; a<edgeGhostsToSend.size() ; ++a )
   {
     if( allNewEdges.count(edgeGhostsToSend[a]) > 0)
     {
       newEdgesToSend.push_back( edgeGhostsToSend[a] );
     }
     if( allModEdges.count(edgeGhostsToSend[a]) > 0)
     {
       modEdgesToSend.push_back( edgeGhostsToSend[a] );
     }
   }

   for( localIndex a=0 ; a<faceGhostsToSend.size() ; ++a )
   {
     if( allNewFaces.count(faceGhostsToSend[a]) > 0)
     {
       newFacesToSend.push_back( faceGhostsToSend[a] );
     }
     if( allModFaces.count(faceGhostsToSend[a]) > 0)
     {
       modFacesToSend.push_back( faceGhostsToSend[a] );
     }
   }

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
           modElemsToSend[er][esr] = elementGhostToSend[er][esr][a];
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

  buffer_type const & receiveBuffer = neighbor->ReceiveBuffer( commID );
  buffer_unit_type const * receiveBufferPtr = receiveBuffer.data();

  localIndex_array newGhostNodes;
  localIndex_array newGhostEdges;
  localIndex_array newGhostFaces;

  localIndex_array modGhostNodes;
  localIndex_array modGhostEdges;
  localIndex_array modGhostFaces;

  ElementRegionManager::ElementViewAccessor<localIndex_array> modGhostElems;
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

}


} /* namespace geosx */
