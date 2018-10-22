/*
 * ParallelTopologyChange.cpp
 *
 *  Created on: Sep 20, 2018
 *      Author: settgast1
 */

#include "ParallelTopologyChange.hpp"


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

  const bool pack=true;
  const realT t0=MPI_Wtime();


  // count the number of new objects that are owned by a neighbor, and send that number to the neighbor

  array1d<MPI_Request> mpiSendGlobalRequest( neighbors.size() );
  array1d<MPI_Request> mpiRecvGlobalRequest( neighbors.size() );
  array1d<MPI_Status>  mpiSendGlobalStatus( neighbors.size() );
  array1d<MPI_Status>  mpiRecvGlobalStatus( neighbors.size() );

  array1d<MPI_Request> mpiSendSizeRequest( neighbors.size() );
  array1d<MPI_Request> mpiRecvSizeRequest( neighbors.size() );
  array1d<MPI_Status>  mpiSendSizeStatus( neighbors.size() );
  array1d<MPI_Status>  mpiRecvSizeStatus( neighbors.size() );

  array1d<MPI_Request> mpiSendBufferRequest( neighbors.size() );
  array1d<MPI_Request> mpiRecvBufferRequest( neighbors.size() );
  array1d<MPI_Status>  mpiSendBufferStatus( neighbors.size() );
  array1d<MPI_Status>  mpiRecvBufferStatus( neighbors.size() );



  for( unsigned int neighborIndex=0 ; neighborIndex<neighbors.size() ; ++neighborIndex )
  {
    NeighborCommunicator& neighbor = neighbors[neighborIndex];

    neighbor.tempNeighborData.clear();

    if(pack)
    neighbor.PackNewGlobalIndexRequests(modifiedObjects);
    neighbor.SendReceive( neighbor.tempNeighborData.sendGlobalIndexRequests, 3, mpiSendGlobalRequest[neighborIndex],
                          neighbor.tempNeighborData.recvGlobalIndexRequests, 3, mpiRecvGlobalRequest[neighborIndex],
                          CommRegistry::genericComm01 );
  }


  // once we have the number of objects this rank needs to create, we can resize() and return the first new
  // global index to the requesting neighbor.
  const realT t1=MPI_Wtime();

  array1d<MPI_Request> mpiSendNewGlobalRequest( neighbors.size() );
  array1d<MPI_Request> mpiRecvNewGlobalRequest( neighbors.size() );
  array1d<MPI_Status>  mpiSendNewGlobalStatus( neighbors.size() );
  array1d<MPI_Status>  mpiRecvNewGlobalStatus( neighbors.size() );

  set<localIndex> newNodeGlobals;
  set<localIndex> newEdgeGlobals;
  set<localIndex> newFaceGlobals;
  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
  {
    int neighborIndex;
    MPI_Waitany( mpiRecvGlobalRequest.size(), mpiRecvGlobalRequest.data(), &neighborIndex, mpiRecvGlobalStatus.data() );

    NeighborCommunicator& neighbor = neighbors[neighborIndex];

    if(pack)
    neighbor.ProcessNewGlobalIndexRequests( newNodeGlobals, newEdgeGlobals, newFaceGlobals );
    neighbor.SendReceive( neighbor.tempNeighborData.sendFirstNewGlobalIndices, 3, mpiSendNewGlobalRequest[neighborIndex],
                          neighbor.tempNeighborData.recvFirstNewGlobalIndices, 3, mpiRecvNewGlobalRequest[neighborIndex],
                          CommRegistry::genericComm02 );
  }

  // now we can assign global indices based on the first recieved by the neighbor.
  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
  {
    int neighborIndex;
    MPI_Waitany( mpiRecvNewGlobalRequest.size(), mpiRecvNewGlobalRequest.data(), &neighborIndex, mpiRecvNewGlobalStatus.data() );
    NeighborCommunicator& neighbor = neighbors[neighborIndex];
    if(pack)
    neighbor.UnpackNewGlobalIndices(  );

  }



  MPI_Waitall( mpiSendGlobalRequest.size() , mpiSendGlobalRequest.data(), mpiSendGlobalStatus.data() );
  MPI_Waitall( mpiSendNewGlobalRequest.size() , mpiSendNewGlobalRequest.data(), mpiSendNewGlobalStatus.data() );

  const realT t2=MPI_Wtime();
  const realT t3=MPI_Wtime();














  // first we need to send over the new objects to our neighbors. This will consist of all new objects owned by this
  // partition that are ghosts on the neighbor partition, or are owned on the neighbor but a ghost on this partition.



  //******************************
  // 1) for a given processor, send new objects created on this processor and the connectivities to neighbors
  //    that either are a ghost on this processor(owned by neighbor), and the ones that are a ghost on the neighbor.

  // pack the buffers, and send the size of the buffers
  for( unsigned int neighborIndex=0 ; neighborIndex<neighbors.size() ; ++neighborIndex )
  {
    NeighborCommunicator& neighbor = neighbors[neighborIndex];

    neighbor.ResizeSendBuffer(0);

    if(pack)
    {
    // pack the new/modified nodes that are owned by this process
    neighbor.PackTopologyModifications( PhysicalDomainT::FiniteElementNodeManager, modifiedObjects.newNodes,  modifiedObjects.modifiedNodes, true, false );
    // pack the new/modified edges that are owned by the neighbor.
    neighbor.PackTopologyModifications( PhysicalDomainT::FiniteElementEdgeManager, modifiedObjects.newEdges,  modifiedObjects.modifiedEdges, true,  true );
    // pack the new/modified edges that are owned by this process
    neighbor.PackTopologyModifications( PhysicalDomainT::FiniteElementEdgeManager, modifiedObjects.newEdges,  modifiedObjects.modifiedEdges, true,  false );

    // pack the new/modified faces that are owned by the neighbor.
    neighbor.PackTopologyModifications( PhysicalDomainT::FiniteElementFaceManager, modifiedObjects.newFaces,  modifiedObjects.modifiedFaces, true,  true );
    // pack the new/modified faces that are owned by this process
    neighbor.PackTopologyModifications( PhysicalDomainT::FiniteElementFaceManager, modifiedObjects.newFaces,  modifiedObjects.modifiedFaces, true,  false );

    // pack the new/modified elements that are owned by the neighbor.
    neighbor.PackTopologyModifications( PhysicalDomainT::FiniteElementElementManager, std::map< std::string, lSet >(),  modifiedObjects.modifiedElements, true,  true );
    // pack the new/modified elements that are owned by this process
    neighbor.PackTopologyModifications( PhysicalDomainT::FiniteElementElementManager, std::map< std::string, lSet >(),  modifiedObjects.modifiedElements, true,  false );
    }
    neighbor.SendReceiveBufferSizes(CommRegistry::genericComm01, mpiSendSizeRequest[neighborIndex], mpiRecvSizeRequest[neighborIndex] );
  }

  // send/recv the buffers
  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
  {
    int neighborIndex;
    MPI_Waitany( mpiRecvSizeRequest.size(), mpiRecvSizeRequest.data(), &neighborIndex, mpiRecvSizeStatus.data() );

    NeighborCommunicator& neighbor = neighbors[neighborIndex];

    neighbor.SendReceiveBuffers( CommRegistry::genericComm01, CommRegistry::genericComm02, mpiSendBufferRequest[neighborIndex], mpiRecvBufferRequest[neighborIndex] );
  }





  lSet allNewAndModifiedLocalNodes;

  lSet allNewNodes, allModifiedNodes;
  lSet allNewEdges, allModifiedEdges;
  lSet allNewFaces, allModifiedFaces;
  std::map< std::string, lSet> allModifiedElements;
  // unpack the buffers
  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
  {
    lArray1d newLocalNodes, modifiedLocalNodes;
    lArray1d newGhostNodes, modifiedGhostNodes;

    lArray1d newLocalEdges, modifiedLocalEdges;
    lArray1d newGhostEdges, modifiedGhostEdges;

    lArray1d newLocalFaces, modifiedLocalFaces;
    lArray1d newGhostFaces, modifiedGhostFaces;

    std::map< std::string, lArray1d> modifiedElements;



    int neighborIndex;
    MPI_Waitany( mpiRecvBufferRequest.size(), mpiRecvBufferRequest.data(), &neighborIndex, mpiRecvBufferStatus.data() );

    NeighborCommunicator& neighbor = this->neighbors[neighborIndex];

    const char* pbuffer = neighbor.ReceiveBuffer().data();

    if(pack)
    {
    neighbor.UnpackTopologyModifications( PhysicalDomainT::FiniteElementNodeManager, pbuffer, newGhostNodes, modifiedGhostNodes, false );
    neighbor.UnpackTopologyModifications( PhysicalDomainT::FiniteElementEdgeManager, pbuffer, newLocalEdges, modifiedLocalEdges, true );
    neighbor.UnpackTopologyModifications( PhysicalDomainT::FiniteElementEdgeManager, pbuffer, newGhostEdges, modifiedGhostEdges, false );
    neighbor.UnpackTopologyModifications( PhysicalDomainT::FiniteElementFaceManager, pbuffer, newLocalFaces, modifiedLocalFaces, true );
    neighbor.UnpackTopologyModifications( PhysicalDomainT::FiniteElementFaceManager, pbuffer, newGhostFaces, modifiedGhostFaces, false );
    neighbor.UnpackTopologyModifications( PhysicalDomainT::FiniteElementElementManager, pbuffer, modifiedElements );
    neighbor.UnpackTopologyModifications( PhysicalDomainT::FiniteElementElementManager, pbuffer, modifiedElements );
    }
    allNewNodes.insert( newLocalNodes.begin(), newLocalNodes.end() );
    allNewNodes.insert( newGhostNodes.begin(), newGhostNodes.end() );

    allModifiedNodes.insert( modifiedLocalNodes.begin(), modifiedLocalNodes.end() );
    allModifiedNodes.insert( modifiedGhostNodes.begin(), modifiedGhostNodes.end() );

    allNewAndModifiedLocalNodes.insert( newLocalNodes.begin(), newLocalNodes.end() );
    allNewAndModifiedLocalNodes.insert( modifiedLocalNodes.begin(), modifiedLocalNodes.end() );

    allNewEdges.insert( newLocalEdges.begin(), newLocalEdges.end() );
    allNewEdges.insert( newGhostEdges.begin(), newGhostEdges.end() );

    allModifiedEdges.insert( modifiedLocalEdges.begin(), modifiedLocalEdges.end() );
    allModifiedEdges.insert( modifiedGhostEdges.begin(), modifiedGhostEdges.end() );

    allNewFaces.insert( newLocalFaces.begin(), newLocalFaces.end() );
    allNewFaces.insert( newGhostFaces.begin(), newGhostFaces.end() );

    allModifiedFaces.insert( modifiedLocalFaces.begin(), modifiedLocalFaces.end() );
    allModifiedFaces.insert( modifiedGhostFaces.begin(), modifiedGhostFaces.end() );

    for( std::map< std::string, lArray1d>::const_iterator i=modifiedElements.begin() ; i!=modifiedElements.end() ; ++i )
    {
      allModifiedElements[i->first].insert( i->second.begin(), i->second.end() );
    }
  }


  MPI_Waitall( mpiSendSizeRequest.size() , mpiSendSizeRequest.data(), mpiSendSizeStatus.data() );
  MPI_Waitall( mpiSendBufferRequest.size() , mpiSendBufferRequest.data(), mpiSendBufferStatus.data() );














  // must send the new objects that are local to this processor, but created on a neighbor, to the other neighbors.
  for( unsigned int neighborIndex=0 ; neighborIndex<neighbors.size() ; ++neighborIndex )
  {
    NeighborCommunicator& neighbor = neighbors[neighborIndex];

    neighbor.ResizeSendBuffer(0);

    if(pack)
    {
    neighbor.PackTopologyModifications( PhysicalDomainT::FiniteElementEdgeManager, allNewEdges,  allModifiedEdges, false, false );
    neighbor.PackTopologyModifications( PhysicalDomainT::FiniteElementFaceManager, allNewFaces,  allModifiedFaces, false, false );
    neighbor.PackTopologyModifications( PhysicalDomainT::FiniteElementElementManager, std::map< std::string, lSet >(),  allModifiedElements, false, false );
    }
    neighbor.SendReceiveBufferSizes(CommRegistry::genericComm01, mpiSendSizeRequest[neighborIndex], mpiRecvSizeRequest[neighborIndex] );
  }

  // send/recv the buffers
  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
  {
    int neighborIndex;
    MPI_Waitany( mpiRecvSizeRequest.size(), mpiRecvSizeRequest.data(), &neighborIndex, mpiRecvSizeStatus.data() );

    NeighborCommunicator& neighbor = neighbors[neighborIndex];

    neighbor.SendReceiveBuffers( CommRegistry::genericComm01, CommRegistry::genericComm02, mpiSendBufferRequest[neighborIndex], mpiRecvBufferRequest[neighborIndex] );
  }

  // unpack the buffers
  for( unsigned int count=0 ; count<neighbors.size() ; ++count )
  {
    lArray1d newLocalNodes, modifiedLocalNodes;
    lArray1d newGhostNodes, modifiedGhostNodes;

    lArray1d newLocalEdges, modifiedLocalEdges;
    lArray1d newGhostEdges, modifiedGhostEdges;

    lArray1d newLocalFaces, modifiedLocalFaces;
    lArray1d newGhostFaces, modifiedGhostFaces;

    std::map< std::string, lArray1d> modifiedElements;



    int neighborIndex;
    MPI_Waitany( mpiRecvBufferRequest.size(), mpiRecvBufferRequest.data(), &neighborIndex, mpiRecvBufferStatus.data() );

    NeighborCommunicator& neighbor = this->neighbors[neighborIndex];

    const char* pbuffer = neighbor.ReceiveBuffer().data();

    if(pack)
    {
      neighbor.UnpackTopologyModifications(PhysicalDomainT::FiniteElementEdgeManager, pbuffer,
                                           newGhostEdges, modifiedGhostEdges, false);
      neighbor.UnpackTopologyModifications(PhysicalDomainT::FiniteElementFaceManager, pbuffer,
                                           newGhostFaces, modifiedGhostFaces, false);
      neighbor.UnpackTopologyModifications(PhysicalDomainT::FiniteElementElementManager, pbuffer,
                                           modifiedElements);
    }
    allNewNodes.insert( newLocalNodes.begin(), newLocalNodes.end() );
    allNewNodes.insert( newGhostNodes.begin(), newGhostNodes.end() );

    allModifiedNodes.insert( modifiedLocalNodes.begin(), modifiedLocalNodes.end() );
    allModifiedNodes.insert( modifiedGhostNodes.begin(), modifiedGhostNodes.end() );

    allNewAndModifiedLocalNodes.insert( newLocalNodes.begin(), newLocalNodes.end() );
    allNewAndModifiedLocalNodes.insert( modifiedLocalNodes.begin(), modifiedLocalNodes.end() );

    allNewEdges.insert( newLocalEdges.begin(), newLocalEdges.end() );
    allNewEdges.insert( newGhostEdges.begin(), newGhostEdges.end() );

    allModifiedEdges.insert( modifiedLocalEdges.begin(), modifiedLocalEdges.end() );
    allModifiedEdges.insert( modifiedGhostEdges.begin(), modifiedGhostEdges.end() );

    allNewFaces.insert( newLocalFaces.begin(), newLocalFaces.end() );
    allNewFaces.insert( newGhostFaces.begin(), newGhostFaces.end() );

    allModifiedFaces.insert( modifiedLocalFaces.begin(), modifiedLocalFaces.end() );
    allModifiedFaces.insert( modifiedGhostFaces.begin(), modifiedGhostFaces.end() );

    for( std::map< std::string, lArray1d>::const_iterator i=modifiedElements.begin() ; i!=modifiedElements.end() ; ++i )
    {
      allModifiedElements[i->first].insert( i->second.begin(), i->second.end() );
    }
  }


  MPI_Waitall( mpiSendSizeRequest.size() , mpiSendSizeRequest.data(), mpiSendSizeStatus.data() );
  MPI_Waitall( mpiSendBufferRequest.size() , mpiSendBufferRequest.data(), mpiSendBufferStatus.data() );




  lSet allReceivedNodes;
  allReceivedNodes.insert( allNewNodes.begin(), allNewNodes.end() );
  allReceivedNodes.insert( allModifiedNodes.begin(), allModifiedNodes.end() );

  allReceivedNodes.erase( allNewAndModifiedLocalNodes.begin(), allNewAndModifiedLocalNodes.end() );

  m_domain->m_feNodeManager.ConnectivityFromGlobalToLocal( allNewAndModifiedLocalNodes,
                                                           allReceivedNodes,
                                                           m_domain->m_feFaceManager.m_globalToLocalMap );


  lSet allReceivedEdges;
  allReceivedEdges.insert( allNewEdges.begin(), allNewEdges.end() );
  allReceivedEdges.insert( allModifiedEdges.begin(), allModifiedEdges.end() );

  m_domain->m_feEdgeManager.ConnectivityFromGlobalToLocal( allReceivedEdges,
                                                           m_domain->m_feNodeManager.m_globalToLocalMap,
                                                           m_domain->m_feFaceManager.m_globalToLocalMap );

  lSet allReceivedFaces;
  allReceivedFaces.insert( allNewFaces.begin(), allNewFaces.end() );
  allReceivedFaces.insert( allModifiedFaces.begin(), allModifiedFaces.end() );

  m_domain->m_feFaceManager.ConnectivityFromGlobalToLocal( allReceivedFaces,
                                                           m_domain->m_feNodeManager.m_globalToLocalMap,
                                                           m_domain->m_feEdgeManager.m_globalToLocalMap );


  m_domain->m_feElementManager.ConnectivityFromGlobalToLocal( allModifiedElements,
                                                              m_domain->m_feNodeManager.m_globalToLocalMap,
                                                              m_domain->m_feFaceManager.m_globalToLocalMap );


  m_domain->m_feNodeManager.ModifyNodeToEdgeMapFromSplit( m_domain->m_feEdgeManager,
                                                          allNewEdges,
                                                          allModifiedEdges );

  m_domain->m_feFaceManager.ModifyToFaceMapsFromSplit( allNewFaces,
                                                       allModifiedFaces,
                                                       m_domain->m_feNodeManager,
                                                       m_domain->m_feEdgeManager,
                                                       m_domain->m_externalFaces );

  m_domain->m_feElementManager.ModifyToElementMapsFromSplit( allModifiedElements,
                                                             m_domain->m_feNodeManager,
                                                             m_domain->m_feFaceManager );

  //This is to update the isExternal attribute of tip node/edge that has just become tip because of splitting of a ghost node.
  m_domain->m_feElementManager.UpdateExternalityFromSplit( allModifiedElements,
                                                           m_domain->m_feNodeManager,
                                                           m_domain->m_feEdgeManager,
                                                           m_domain->m_feFaceManager);
//
//  m_domain->m_feEdgeManager.UpdateEdgeExternalityFromSplit( m_domain->m_feFaceManager,
//                                                            allNewEdges,
//                                                            allModifiedEdges);


  const realT t4=MPI_Wtime();



//  m_t1 += t1-t0;
//  m_t2 += t2-t1;
//  m_t3 += t3-t2;
//  m_t4 += t4-t0;




}

} /* namespace lvarray */
