/*
 * ParallelTopologyChange.cpp
 *
 *  Created on: Sep 20, 2018
 *      Author: settgast1
 */

#include "ParallelTopologyChange.hpp"
#include "MPI_Communications/CommunicationTools.hpp"
#include "mesh/ElementRegionManager.hpp"

namespace geosx
{

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
  ElementRegionManager * const elemRegManager = mesh->getElemManager();

  array1d<integer> & nodeGhostRank = nodeManager->GhostRank();
  array1d<integer> & edgeGhostRank = edgeManager->GhostRank();
  array1d<integer> & faceGhostRank = faceManager->GhostRank();

  // 1) Assign new global indices to the new objects
  CommunicationTools::AssignNewGlobalIndices( *nodeManager, modifiedObjects.newNodes );
  CommunicationTools::AssignNewGlobalIndices( *edgeManager, modifiedObjects.newEdges );
  CommunicationTools::AssignNewGlobalIndices( *faceManager, modifiedObjects.newFaces );




  // 2) first we need to send over the new objects to owning ranks. New objects are assumed to be
  // owned by the rank that owned the parent.

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
    modElemData.resize(elemRegManager->numRegions());
    for( localIndex er=0 ; er<elemRegManager->numRegions() ; ++er )
    {
      ElementRegion * const elemRegion = elemRegManager->GetRegion(er);
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


    neighbor.PackNewAndModifiedObjects( mesh,
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


  // send/recv the buffers
  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
  {
    int neighborIndex;
    MPI_Waitany( commData.size,
                 commData.mpiRecvBufferRequest.data(),
                 &neighborIndex,
                 commData.mpiRecvBufferStatus.data() );

    NeighborCommunicator& neighbor = neighbors[neighborIndex];


  }



  MPI_Waitall( commData.size,
               commData.mpiSizeSendBufferRequest.data(),
               commData.mpiSizeSendBufferStatus.data() );


//
//  lSet allNewAndModifiedLocalNodes;
//
//  lSet allNewNodes, allModifiedNodes;
//  lSet allNewEdges, allModifiedEdges;
//  lSet allNewFaces, allModifiedFaces;
//  std::map< std::string, lSet> allModifiedElements;
//  // unpack the buffers
//  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
//  {
//    lArray1d newLocalNodes, modifiedLocalNodes;
//    lArray1d newGhostNodes, modifiedGhostNodes;
//
//    lArray1d newLocalEdges, modifiedLocalEdges;
//    lArray1d newGhostEdges, modifiedGhostEdges;
//
//    lArray1d newLocalFaces, modifiedLocalFaces;
//    lArray1d newGhostFaces, modifiedGhostFaces;
//
//    std::map< std::string, lArray1d> modifiedElements;
//
//
//
//    int neighborIndex;
//    MPI_Waitany( mpiRecvBufferRequest.size(), mpiRecvBufferRequest.data(), &neighborIndex, mpiRecvBufferStatus.data() );
//
//    NeighborCommunicator& neighbor = this->neighbors[neighborIndex];
//
//    const char* pbuffer = neighbor.ReceiveBuffer().data();
//
//    if(pack)
//    {
//    neighbor.UnpackTopologyModifications( PhysicalDomainT::FiniteElementNodeManager, pbuffer, newGhostNodes, modifiedGhostNodes, false );
//    neighbor.UnpackTopologyModifications( PhysicalDomainT::FiniteElementEdgeManager, pbuffer, newLocalEdges, modifiedLocalEdges, true );
//    neighbor.UnpackTopologyModifications( PhysicalDomainT::FiniteElementEdgeManager, pbuffer, newGhostEdges, modifiedGhostEdges, false );
//    neighbor.UnpackTopologyModifications( PhysicalDomainT::FiniteElementFaceManager, pbuffer, newLocalFaces, modifiedLocalFaces, true );
//    neighbor.UnpackTopologyModifications( PhysicalDomainT::FiniteElementFaceManager, pbuffer, newGhostFaces, modifiedGhostFaces, false );
//    neighbor.UnpackTopologyModifications( PhysicalDomainT::FiniteElementElementManager, pbuffer, modifiedElements );
//    neighbor.UnpackTopologyModifications( PhysicalDomainT::FiniteElementElementManager, pbuffer, modifiedElements );
//    }
//    allNewNodes.insert( newLocalNodes.begin(), newLocalNodes.end() );
//    allNewNodes.insert( newGhostNodes.begin(), newGhostNodes.end() );
//
//    allModifiedNodes.insert( modifiedLocalNodes.begin(), modifiedLocalNodes.end() );
//    allModifiedNodes.insert( modifiedGhostNodes.begin(), modifiedGhostNodes.end() );
//
//    allNewAndModifiedLocalNodes.insert( newLocalNodes.begin(), newLocalNodes.end() );
//    allNewAndModifiedLocalNodes.insert( modifiedLocalNodes.begin(), modifiedLocalNodes.end() );
//
//    allNewEdges.insert( newLocalEdges.begin(), newLocalEdges.end() );
//    allNewEdges.insert( newGhostEdges.begin(), newGhostEdges.end() );
//
//    allModifiedEdges.insert( modifiedLocalEdges.begin(), modifiedLocalEdges.end() );
//    allModifiedEdges.insert( modifiedGhostEdges.begin(), modifiedGhostEdges.end() );
//
//    allNewFaces.insert( newLocalFaces.begin(), newLocalFaces.end() );
//    allNewFaces.insert( newGhostFaces.begin(), newGhostFaces.end() );
//
//    allModifiedFaces.insert( modifiedLocalFaces.begin(), modifiedLocalFaces.end() );
//    allModifiedFaces.insert( modifiedGhostFaces.begin(), modifiedGhostFaces.end() );
//
//    for( std::map< std::string, lArray1d>::const_iterator i=modifiedElements.begin() ; i!=modifiedElements.end() ; ++i )
//    {
//      allModifiedElements[i->first].insert( i->second.begin(), i->second.end() );
//    }
//  }
//
//
//  MPI_Waitall( mpiSendSizeRequest.size() , mpiSendSizeRequest.data(), mpiSendSizeStatus.data() );
//  MPI_Waitall( mpiSendBufferRequest.size() , mpiSendBufferRequest.data(), mpiSendBufferStatus.data() );
//
//


//
//
//
//
//
//
//
//
//
//
//  // must send the new objects that are local to this processor, but created on a neighbor, to the other neighbors.
//  for( unsigned int neighborIndex=0 ; neighborIndex<neighbors.size() ; ++neighborIndex )
//  {
//    NeighborCommunicator& neighbor = neighbors[neighborIndex];
//
//    neighbor.ResizeSendBuffer(0);
//
//    if(pack)
//    {
//    neighbor.PackTopologyModifications( PhysicalDomainT::FiniteElementEdgeManager, allNewEdges,  allModifiedEdges, false, false );
//    neighbor.PackTopologyModifications( PhysicalDomainT::FiniteElementFaceManager, allNewFaces,  allModifiedFaces, false, false );
//    neighbor.PackTopologyModifications( PhysicalDomainT::FiniteElementElementManager, std::map< std::string, lSet >(),  allModifiedElements, false, false );
//    }
//    neighbor.SendReceiveBufferSizes(CommRegistry::genericComm01, mpiSendSizeRequest[neighborIndex], mpiRecvSizeRequest[neighborIndex] );
//  }
//
//  // send/recv the buffers
//  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
//  {
//    int neighborIndex;
//    MPI_Waitany( mpiRecvSizeRequest.size(), mpiRecvSizeRequest.data(), &neighborIndex, mpiRecvSizeStatus.data() );
//
//    NeighborCommunicator& neighbor = neighbors[neighborIndex];
//
//    neighbor.SendReceiveBuffers( CommRegistry::genericComm01, CommRegistry::genericComm02, mpiSendBufferRequest[neighborIndex], mpiRecvBufferRequest[neighborIndex] );
//  }
//
//  // unpack the buffers
//  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
//  {
//    lArray1d newLocalNodes, modifiedLocalNodes;
//    lArray1d newGhostNodes, modifiedGhostNodes;
//
//    lArray1d newLocalEdges, modifiedLocalEdges;
//    lArray1d newGhostEdges, modifiedGhostEdges;
//
//    lArray1d newLocalFaces, modifiedLocalFaces;
//    lArray1d newGhostFaces, modifiedGhostFaces;
//
//    std::map< std::string, lArray1d> modifiedElements;
//
//
//
//    int neighborIndex;
//    MPI_Waitany( mpiRecvBufferRequest.size(), mpiRecvBufferRequest.data(), &neighborIndex, mpiRecvBufferStatus.data() );
//
//    NeighborCommunicator& neighbor = this->neighbors[neighborIndex];
//
//    const char* pbuffer = neighbor.ReceiveBuffer().data();
//
//    if(pack)
//    {
//      neighbor.UnpackTopologyModifications(PhysicalDomainT::FiniteElementEdgeManager, pbuffer,
//                                           newGhostEdges, modifiedGhostEdges, false);
//      neighbor.UnpackTopologyModifications(PhysicalDomainT::FiniteElementFaceManager, pbuffer,
//                                           newGhostFaces, modifiedGhostFaces, false);
//      neighbor.UnpackTopologyModifications(PhysicalDomainT::FiniteElementElementManager, pbuffer,
//                                           modifiedElements);
//    }
//    allNewNodes.insert( newLocalNodes.begin(), newLocalNodes.end() );
//    allNewNodes.insert( newGhostNodes.begin(), newGhostNodes.end() );
//
//    allModifiedNodes.insert( modifiedLocalNodes.begin(), modifiedLocalNodes.end() );
//    allModifiedNodes.insert( modifiedGhostNodes.begin(), modifiedGhostNodes.end() );
//
//    allNewAndModifiedLocalNodes.insert( newLocalNodes.begin(), newLocalNodes.end() );
//    allNewAndModifiedLocalNodes.insert( modifiedLocalNodes.begin(), modifiedLocalNodes.end() );
//
//    allNewEdges.insert( newLocalEdges.begin(), newLocalEdges.end() );
//    allNewEdges.insert( newGhostEdges.begin(), newGhostEdges.end() );
//
//    allModifiedEdges.insert( modifiedLocalEdges.begin(), modifiedLocalEdges.end() );
//    allModifiedEdges.insert( modifiedGhostEdges.begin(), modifiedGhostEdges.end() );
//
//    allNewFaces.insert( newLocalFaces.begin(), newLocalFaces.end() );
//    allNewFaces.insert( newGhostFaces.begin(), newGhostFaces.end() );
//
//    allModifiedFaces.insert( modifiedLocalFaces.begin(), modifiedLocalFaces.end() );
//    allModifiedFaces.insert( modifiedGhostFaces.begin(), modifiedGhostFaces.end() );
//
//    for( std::map< std::string, lArray1d>::const_iterator i=modifiedElements.begin() ; i!=modifiedElements.end() ; ++i )
//    {
//      allModifiedElements[i->first].insert( i->second.begin(), i->second.end() );
//    }
//  }
//
//
//  MPI_Waitall( mpiSendSizeRequest.size() , mpiSendSizeRequest.data(), mpiSendSizeStatus.data() );
//  MPI_Waitall( mpiSendBufferRequest.size() , mpiSendBufferRequest.data(), mpiSendBufferStatus.data() );
//
//
//
//
//  lSet allReceivedNodes;
//  allReceivedNodes.insert( allNewNodes.begin(), allNewNodes.end() );
//  allReceivedNodes.insert( allModifiedNodes.begin(), allModifiedNodes.end() );
//
//  allReceivedNodes.erase( allNewAndModifiedLocalNodes.begin(), allNewAndModifiedLocalNodes.end() );
//
//  m_domain->m_feNodeManager.ConnectivityFromGlobalToLocal( allNewAndModifiedLocalNodes,
//                                                           allReceivedNodes,
//                                                           m_domain->m_feFaceManager.m_globalToLocalMap );
//
//
//  lSet allReceivedEdges;
//  allReceivedEdges.insert( allNewEdges.begin(), allNewEdges.end() );
//  allReceivedEdges.insert( allModifiedEdges.begin(), allModifiedEdges.end() );
//
//  m_domain->m_feEdgeManager.ConnectivityFromGlobalToLocal( allReceivedEdges,
//                                                           m_domain->m_feNodeManager.m_globalToLocalMap,
//                                                           m_domain->m_feFaceManager.m_globalToLocalMap );
//
//  lSet allReceivedFaces;
//  allReceivedFaces.insert( allNewFaces.begin(), allNewFaces.end() );
//  allReceivedFaces.insert( allModifiedFaces.begin(), allModifiedFaces.end() );
//
//  m_domain->m_feFaceManager.ConnectivityFromGlobalToLocal( allReceivedFaces,
//                                                           m_domain->m_feNodeManager.m_globalToLocalMap,
//                                                           m_domain->m_feEdgeManager.m_globalToLocalMap );
//
//
//  m_domain->m_feElementManager.ConnectivityFromGlobalToLocal( allModifiedElements,
//                                                              m_domain->m_feNodeManager.m_globalToLocalMap,
//                                                              m_domain->m_feFaceManager.m_globalToLocalMap );
//
//
//  m_domain->m_feNodeManager.ModifyNodeToEdgeMapFromSplit( m_domain->m_feEdgeManager,
//                                                          allNewEdges,
//                                                          allModifiedEdges );
//
//  m_domain->m_feFaceManager.ModifyToFaceMapsFromSplit( allNewFaces,
//                                                       allModifiedFaces,
//                                                       m_domain->m_feNodeManager,
//                                                       m_domain->m_feEdgeManager,
//                                                       m_domain->m_externalFaces );
//
//  m_domain->m_feElementManager.ModifyToElementMapsFromSplit( allModifiedElements,
//                                                             m_domain->m_feNodeManager,
//                                                             m_domain->m_feFaceManager );
//
//  //This is to update the isExternal attribute of tip node/edge that has just become tip because of splitting of a ghost node.
//  m_domain->m_feElementManager.UpdateExternalityFromSplit( allModifiedElements,
//                                                           m_domain->m_feNodeManager,
//                                                           m_domain->m_feEdgeManager,
//                                                           m_domain->m_feFaceManager);
////
////  m_domain->m_feEdgeManager.UpdateEdgeExternalityFromSplit( m_domain->m_feFaceManager,
////                                                            allNewEdges,
////                                                            allModifiedEdges);
//
//
//  const realT t4=MPI_Wtime();
//
//
//
////  m_t1 += t1-t0;
////  m_t2 += t2-t1;
////  m_t3 += t3-t2;
////  m_t4 += t4-t0;




}

} /* namespace geosx */
