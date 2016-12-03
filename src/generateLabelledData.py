from generateNetwork import *
import numpy as np

# date, time, nodeid, epoch, temp(0, 50), humidity(0,100), light(1, 100000), voltage(2,3)

createAllGraphs('../data/mote_locs.txt', '../data/connectivity.txt', '../data/data.txt')
print 'Preprocess done'
graph_0 = getGraphAtEpoch(60000)

nNodes = graph_0.GetNodes()
feats_0 = np.zeros((nNodes, 4))
for node in graph_0.Nodes():
    nid = node.GetId()
    feats_0[nid-1][:] = np.array(getNodeFeatures(nid, graph_0))

limits = [(0.0,50.0), (0.0,100.0), (1.0, 100000.0), (2.0,3.0)]
nEpochs = 65535

anomalyProb = 0.001
anomalies = []

with open('../data/fakedata.txt', 'wb') as f1:
    lastFeats = feats_0
    for epoch in xrange(1, nEpochs+1):
        newFeats = lastFeats.copy()
        newFeatsWoAnomalies= lastFeats.copy()
        for nid in xrange(nNodes):
            for feat in xrange(4):
                diff = 0.005*(limits[feat][1] - limits[feat][0])
                delta = np.random.uniform(0, 2*diff) - diff
                randProb = np.random.uniform()
                if randProb < anomalyProb:
                    t= np.random.uniform(limits[feat][0], limits[feat][1])
                    newFeats[nid, feat] = t
                    anomalies.append((epoch, nid, feat))
                else:
                    newFeats[nid, feat] += delta
                    newFeatsWoAnomalies[nid, feat] = newFeats[nid, feat]
                    newFeats[nid, feat] = max(limits[feat][0], newFeats[nid, feat])
                    newFeats[nid, feat] = min(limits[feat][1], newFeats[nid, feat])

            line = ['00:00', '00:00', epoch, nid+1]
            line.extend(newFeats[nid, :].tolist())
            line = [str(x) for x in line]
            f1.writelines([' '.join(line)])
            f1.write('\n')
        lastFeats = newFeatsWoAnomalies

with open('../data/fakedatalabels.txt', 'wb') as f1:
    f1.writelines(['epoch nodeid featureid\n'])
    for anomaly in anomalies:
        f1.writelines([' '.join([str(_) for _ in anomaly])])
        f1.write('\n')

print len(anomalies)
