from FeatGraph import *
from generateNetwork import *
import numpy as np

def getDnodaFeats(nid, epoch):
    graph = getGraphAtEpoch(epoch)
    node = graph.GetNI(nid)
    nbrs = [node.GetOutNId(i) for i in xrange(node.GetOutDeg())]
    if epoch > 1:
        graph_tm1 = getGraphAtEpoch(epoch - 1)
    else:
        graph_tm1 = graph
    if epoch < max_epoch:
        graph_tp1 = getGraphAtEpoch(epoch + 1)
    else:
        graph_tp1 = graph

    allFeats = getNodeFeatures(nid, graph)
    allFeats.extend(getNodeFeatures(nid, graph_tm1))
    allFeats.extend(getNodeFeatures(nid, graph_tp1))
    nbrFeats = []
    nbrFeats.extend([getNodeFeatures(nbrid, graph) for nbrid in nbrs])
    nbrFeats = np.array(nbrFeats)
    if nbrFeats.shape[0] == 0:
        allFeats.extend(getNodeFeatures(nid, graph))
    else:
        nbrFeats = np.sum(nbrFeats, axis = 0)/(nbrFeats.shape[0]*1.0)
        allFeats.extend(nbrFeats.tolist())
    return allFeats

createAllGraphs('../data/mote_locs.txt', '../data/connectivity.txt', '../data/data.txt')
with open('../data/dnodaFeats.txt', 'wb') as f1:
    for epoch in xrange(1, 65535):
        #print epoch
        for nid in xrange(1,55):
            feats = getDnodaFeats(nid, epoch)
            f1.writelines([' '.join([str(x) for x in feats])])
            f1.write('\n')
