package org.tmu;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.clustering.MultiKMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.tmu.clustering.SquaredEuclideanDistance;
import org.tmu.clustering.WeightedPoint;
import org.tmu.util.CSVReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 9/20/13
 * Time: 10:01 PM
 * To change this template use File | Settings | File Templates.
 */
public class Main {
    public static void main(String[] args) throws IOException {
        List<DoublePoint> points=new ArrayList<DoublePoint>();

        CSVReader csvReader=new CSVReader("X:\\DataSets\\sets\\s4-15.txt");
        int point_size=-1;
        int i=0;
        while (true){
            double[] vec=csvReader.ReadNextDoubleVector();
            if(vec==null)
                break;
            if(point_size==-1)
                point_size=vec.length;
            if(point_size!=vec.length)
                continue;
//            if(i++>1000000)
//                break;
            points.add(new DoublePoint(vec));
        }

        //System.in.read();

        long t0=System.nanoTime();
        KMeansPlusPlusClusterer<DoublePoint> kmeans2=new KMeansPlusPlusClusterer<DoublePoint>(15,20,new EuclideanDistance());
        MultiKMeansPlusPlusClusterer<DoublePoint> kmeans=new MultiKMeansPlusPlusClusterer<DoublePoint>(kmeans2,30);
        List<CentroidCluster<DoublePoint>> clusters= kmeans.cluster(points);
        List<DoublePoint> centers=new ArrayList<DoublePoint>();
        for(CentroidCluster<DoublePoint> c:clusters)
            centers.add(new DoublePoint(c.getCenter().getPoint()));

        double sse=0;
        EuclideanDistance euclideanDistance=new EuclideanDistance();
        for(DoublePoint point:points){
            double distance=Double.MAX_VALUE;
            for(DoublePoint center:centers)
                if(distance> euclideanDistance.compute(center.getPoint(),point.getPoint()))
                    distance=euclideanDistance.compute(center.getPoint(),point.getPoint());
            sse+=distance*distance;
        }

        System.out.println("sse: "+sse);
        System.out.printf("time: %,d\n", System.nanoTime() - t0);


    }
}
