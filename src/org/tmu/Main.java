package org.tmu;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.tmu.clustering.Evaluator;
import org.tmu.clustering.MultiKMeansPlusPlus;
import org.tmu.clustering.StreamKMeansPlusPlus;
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
        //parsing command line
        // create the command line parser
        CommandLineParser parser = new BasicParser();
        //System.in.read();

        // create the Options
        Options options = new Options();
        options.addOption("i", "input", true, "the input file name.");
        options.addOption("a", "all", false, "enumerate all subgraphs.");
        options.addOption("silent", false, "suppress progress report.");
        HelpFormatter formatter = new HelpFormatter();


        List<DoublePoint> points = CSVReader.readAllPointsFromFile("X:\\DataSets\\sets\\USCensus1990.txt");
        System.out.printf("points: %,d\n",points.size());


        //System.in.read();
        long t0 = System.nanoTime();

        StreamKMeansPlusPlus streamKMeansPlusPlus=new StreamKMeansPlusPlus("X:\\DataSets\\sets\\USCensus1990.txt");
        List<CentroidCluster<DoublePoint>> clusters=streamKMeansPlusPlus.cluster(10,10000,5);
        System.out.println("++++++++++++++++++++++++++++++");
        System.out.println("sse: " + Evaluator.computeSSE(clusters,points));
        System.out.println("icd: " + Evaluator.computeICD(clusters));
        System.out.printf("time: %,d\n", System.nanoTime() - t0);

        MultiKMeansPlusPlus multiKMeansPlusPlus=new MultiKMeansPlusPlus(10,5);
        clusters=multiKMeansPlusPlus.cluster(points);

//        for (int counter = 0; counter < 7; counter++) {
//
//            KMeansPlusPlusClusterer<DoublePoint> kmeans = new KMeansPlusPlusClusterer<DoublePoint>(15, (int) Math.log(points.size()) * counter, new EuclideanDistance());
//            List<CentroidCluster<DoublePoint>> clusters = kmeans.cluster(points);
            System.out.println("++++++++++++++++++++++++++++++");
            System.out.println("sse: " + Evaluator.computeSSE(clusters));
            System.out.println("icd: " + Evaluator.computeICD(clusters));
            System.out.printf("time: %,d\n", System.nanoTime() - t0);
//
//        }


    }
}
