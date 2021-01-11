from node2vec import Node2Vec # for make vectors from nodes
import networkx as nx # for graph
import numpy as np
import math # for cosinos between vectors
#G.adj[1] neighbor for node 1
# g.edges() - all edges
#g.nodes() - all nodes
def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)
class Data():
    def __init__(self,eSet):
        self.g=nx.Graph()
        for e in eSet:
            if len(e)==3:
                self.g.add_weighted_edges_from([e])
            else : self.g.add_edge(*e)
    
        
    def subGrpahExtractionAlgorithm(self,x,h):
        vhx=set()
        curr_nb=set()
        vhx.add(x)
        curr_nb.add(x)
        for i in range (h):
            if len(curr_nb) == 0:
                break
            for v in curr_nb:
                curr_nb=curr_nb.union(set(self.g.adj[v]))
            curr_nb=curr_nb.difference(vhx)
            vhx=vhx.union(curr_nb)
            g=nx.Graph()
            g.add_edges_from(self.g.edges(vhx))
        return  g
    def sortSubGraph(self,ghx,x):
        node2vec=Node2Vec(ghx, dimensions=20, walk_length=16, num_walks=100, workers=2)
        model=node2vec.fit(window=10, min_count=1)
        nodes = [x for x in model.wv.vocab]
        embeddings = np.array([model.wv[x] for x in nodes])
        node_vec_dist=set()
        for v_idx,v in enumerate(embeddings):
            cal=cosine_similarity(v,embeddings[x])
            node_vec_dist.add((v_idx,cal))
        seq=sorted(node_vec_dist, key=lambda tup: tup[1])
        seq.reverse()
        return seq
    def node_Information_Matrix_Construction(self,ghx,seqX,k):
        adjMatrix=np.zeros((k,k))
        for idx_i,node_i in enumerate(seqX):
            if (idx_i>k-1): break
            connected=ghx.adj[node_i[0]]
            for idx_j,node_j in enumerate(seqX):
                if (idx_j>k-1): break
                if (node_j[0] in connected):
                    adjMatrix[idx_i][idx_j]=1
                else:
                    adjMatrix[idx_i][idx_j]=0
                
        return adjMatrix
    def mergeAndLabel(self,adjX,adjY,x,y,k):
        adjMatrix=np.zeros((k,k,2))
        adjMatrix[:,:,0]=adjX[:,:]
        adjMatrix[:,:,1]=adjY[:,:]
        if y in self.g.adj[x]:
            label=1
        else:
            label=0
        return adjMatrix,label
    def getDataOfPair(self,x,y,k,h):
       ghx=self.subGrpahExtractionAlgorithm(x,h)
       seqX=self.sortSubGraph(ghx,x)
       adjX=self.node_Information_Matrix_Construction(ghx,seqX,k)
       ghy=self.subGrpahExtractionAlgorithm(y,h)
       seqY=self.sortSubGraph(ghy,y)
       adjY=self.node_Information_Matrix_Construction(ghy,seqY,k)
       adj,label=self.mergeAndLabel(adjX,adjY,x,y,k)
       return adj,label
    def getData(self,k=10,h=3):
        data=list()
        for node1 in self.g.nodes():
            for node2 in self.g.nodes():
                if node1 == node2: continue
                data.append(self.getDataOfPair(node1,node2,k,h))
        return data
       
       
def test():
    eTest=set(((0,1),(1,3),(2,1),(4,2),(5,2),(6,5)))
    testData=Data(eTest)
    data=testData.getData()
    return data
    
#data=test()