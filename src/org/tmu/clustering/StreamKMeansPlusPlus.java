package org.tmu.clustering;

import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.Clusterer;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.tmu.util.CSVReader;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 9/22/13
 * Time: 2:56 PM
 * To change this template use File | Settings | File Templates.
 */
public class StreamKMeansPlusPlus{
    boolean verbose=true;
    String path;
    List<ClusterPoint> intermediateClusterPoints=new ArrayList<ClusterPoint>();

    public StreamKMeansPlusPlus(String path){
        this.path=path;
    }

    public List<CentroidCluster<DoublePoint>> cluster(int k,int chunk_size, int chunk_iters) throws IOException {
        CSVReader csvReader=new CSVReader(path);
        List<DoublePoint> points;
        int chunk_number=0;
        do{
            points=csvReader.readNextPoints(chunk_size);
            if(points.size()==0)
                break;
            MultiKMeansPlusPlus multiKMeansPlusPlus=new MultiKMeansPlusPlus(k,chunk_iters);
            List<CentroidCluster<DoublePoint>> clusters=multiKMeansPlusPlus.cluster(points);
            for(CentroidCluster<DoublePoint> cluster:clusters)
                intermediateClusterPoints.add(new ClusterPoint(cluster));
            if(verbose)
                System.out.printf("Chunk number: %,d\n",chunk_number++);
        }while (points.size()>0);

        List<DoublePoint> centers=new ArrayList<DoublePoint>(intermediateClusterPoints.size());
        for(ClusterPoint cp:intermediateClusterPoints)
            centers.add(new DoublePoint(cp.center));

        csvReader.close();
        return new MultiKMeansPlusPlus(k,5).cluster(centers);
    }
}
