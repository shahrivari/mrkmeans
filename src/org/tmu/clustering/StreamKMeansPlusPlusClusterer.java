package org.tmu.clustering;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.tmu.util.CSVReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 9/22/13
 * Time: 2:56 PM
 * To change this template use File | Settings | File Templates.
 */
public class StreamKMeansPlusPlusClusterer {
    public boolean verbose = false;
    String path;
    List<ClusterPoint> intermediateClusterPoints = new ArrayList<ClusterPoint>();

    public StreamKMeansPlusPlusClusterer(String path) {
        this.path = path;
    }

    public List<CentroidCluster<DoublePoint>> cluster(int k, int chunk_size, int tries) throws IOException {
        int klogk = k * (int) (Math.log(k) + 1) * 3;
        if (verbose) {
            System.out.println("The chunk size is: " + chunk_size);
            System.out.println("Tries for intermediate: " + tries);
            System.out.println("Centers per chunk: " + klogk);
        }
        CSVReader csvReader = new CSVReader(path);
        List<DoublePoint> points;

        do {
            points = csvReader.readNextPoints(chunk_size);
            if (points.size() == 0 || points.size() < k)
                break;
            MultiKMeansPlusPlus multiKMeansPlusPlus = new MultiKMeansPlusPlus(klogk, 1, tries);
            List<CentroidCluster<DoublePoint>> clusters = multiKMeansPlusPlus.cluster(points);
            for (CentroidCluster<DoublePoint> cluster : clusters)
                intermediateClusterPoints.add(new ClusterPoint(cluster));
            if (verbose)
                System.out.printf(".");
        } while (points.size() > 0);

        if (verbose)
            System.out.println();

        List<DoublePoint> centers = new ArrayList<DoublePoint>(intermediateClusterPoints.size());
        for (ClusterPoint cp : intermediateClusterPoints)
            centers.add(new DoublePoint(cp.center));

        csvReader.close();
        if (verbose)
            System.out.println("The intermediate centers size is: " + centers.size());
        MultiKMeansPlusPlus last = new MultiKMeansPlusPlus(k, (int) Math.log(centers.size()) * 2, 3);
        last.verbose = verbose;
        return last.cluster(centers);
    }
}
