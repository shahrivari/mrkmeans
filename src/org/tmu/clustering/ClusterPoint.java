package org.tmu.clustering;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.distance.DistanceMeasure;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 9/22/13
 * Time: 2:44 PM
 * To change this template use File | Settings | File Templates.
 */
public class ClusterPoint {
    int count;
    double sse;
    double[] center;

    public ClusterPoint(CentroidCluster<DoublePoint> cluster) {
        count = cluster.getPoints().size();
        center = cluster.getCenter().getPoint();
        sse = Evaluator.computeSSEofCluster(cluster);
    }
}
