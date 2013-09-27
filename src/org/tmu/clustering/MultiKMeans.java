package org.tmu.clustering;

import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Clusterer;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import java.util.Collection;
import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 9/22/13
 * Time: 12:51 PM
 * To change this template use File | Settings | File Templates.
 */
public class MultiKMeans extends Clusterer<DoublePoint> {
    int k;
    int iterations=10;
    int tries=3;
    public boolean verbose=false;

    protected MultiKMeans(DistanceMeasure measure) {
        super(measure);
    }

    public MultiKMeans(int k){
        super(new EuclideanDistance());
        this.k=k;
    }

    public MultiKMeans(int k, int iterations,int tries){
        super(new EuclideanDistance());
        this.iterations=iterations;
        this.k=k;
        this.tries=tries;
    }

    @Override
    public List<CentroidCluster<DoublePoint>> cluster(Collection<DoublePoint> points) throws MathIllegalArgumentException, ConvergenceException {
        List<CentroidCluster<DoublePoint>> bestClusters=null;
        double bestSSE=Double.MAX_VALUE;

        for(int i=0;i<tries;i++){
            //int max_iterations=Math.max((int) Math.log(points.size()) * iterations, 1);
            KMeansClusterer<DoublePoint> kMeansClusterer=new KMeansClusterer<DoublePoint>(k,iterations);
            List<CentroidCluster<DoublePoint>> clusters=kMeansClusterer.cluster(points);
            double sse=Evaluator.computeSSE(clusters);
            if(bestSSE>sse){
                bestClusters=clusters;
                bestSSE=sse;
            }
            if(verbose)
                System.out.printf("sse: %e\t iters: %d\n",sse,iterations);
        }
        return bestClusters;
    }


}
