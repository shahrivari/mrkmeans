package org.tmu.clustering;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.tmu.util.CSVReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 9/22/13
 * Time: 12:19 PM
 * To change this template use File | Settings | File Templates.
 */
public class Evaluator {

    public static double computeSSE(List<CentroidCluster<DoublePoint>> clusters) {
        double sse = 0;
        EuclideanDistance euclideanDistance = new EuclideanDistance();
        for (CentroidCluster<DoublePoint> cluster : clusters) {
            Clusterable center = cluster.getCenter();
            for (DoublePoint point : cluster.getPoints()) {
                double distance = euclideanDistance.compute(center.getPoint(), point.getPoint());
                sse += distance * distance;
            }
        }
        return sse;
    }

    public static double computeSSE(List<CentroidCluster<DoublePoint>> clusters, Collection<DoublePoint> points) {
        double sse = 0;
        EuclideanDistance euclideanDistance = new EuclideanDistance();
        for (DoublePoint point : points) {
            double best_distance = Double.MAX_VALUE;
            for (CentroidCluster<DoublePoint> cluster : clusters) {
                Clusterable center = cluster.getCenter();
                double distance = euclideanDistance.compute(center.getPoint(), point.getPoint());
                if (distance < best_distance)
                    best_distance = distance;
            }
            sse += best_distance * best_distance;
        }
        return sse;
    }

    public static double computeSSE(List<CentroidCluster<DoublePoint>> clusters, String file_path) throws IOException {
        double sse = 0;
        EuclideanDistance euclideanDistance = new EuclideanDistance();
        CSVReader csvReader = new CSVReader(file_path);
        List<DoublePoint> points = csvReader.readNextPoints(1000);
        while (points.size() > 0) {
            for (DoublePoint point : points) {
                double best_distance = Double.MAX_VALUE;
                for (CentroidCluster<DoublePoint> cluster : clusters) {
                    Clusterable center = cluster.getCenter();
                    double distance = euclideanDistance.compute(center.getPoint(), point.getPoint());
                    if (distance < best_distance)
                        best_distance = distance;
                }
                sse += best_distance * best_distance;
            }
            points = csvReader.readNextPoints(1000);
        }
        csvReader.close();
        return sse;
    }


    public static double computeSSEofCluster(CentroidCluster<DoublePoint> cluster) {
        double sse = 0;
        EuclideanDistance euclideanDistance = new EuclideanDistance();
        Clusterable center = cluster.getCenter();
        for (DoublePoint point : cluster.getPoints()) {
            double distance = euclideanDistance.compute(center.getPoint(), point.getPoint());
            sse += distance * distance;
        }
        return sse;
    }


    public static double computeICD(List<CentroidCluster<DoublePoint>> clusters) {
        double icd = 0;
        EuclideanDistance euclideanDistance = new EuclideanDistance();
        List<Clusterable> centers = new ArrayList<Clusterable>();

        for (CentroidCluster<DoublePoint> cluster : clusters)
            centers.add(cluster.getCenter());

        for (int i = 0; i < centers.size(); i++)
            for (int j = i + 1; j < centers.size(); j++)
                icd += euclideanDistance.compute(centers.get(i).getPoint(), centers.get(j).getPoint());

        return icd;
    }

}
