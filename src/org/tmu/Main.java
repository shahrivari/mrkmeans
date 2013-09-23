package org.tmu;

import org.apache.commons.cli.*;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.tmu.clustering.*;
import org.tmu.util.CSVReader;

import java.io.IOException;
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
        options.addOption("k", "k", true, "the number of clusters.");
        options.addOption("plus", false, "use kmeans++.");
        options.addOption("standard", false, "use standard kmeans.");
        options.addOption("stream", false, "use stream kmeans++.");
        options.addOption("t", "tries", true, "try x times and return the best. Also denotes the per chunk iteration for the stream case.");
        options.addOption("p", "print", true, "print the final centers.");
        options.addOption("m", "max", true, "the max iterations.");
        options.addOption("c", "chunk", true, "the chunk size.");
        options.addOption("v", "verbose", false, "be verbose.");

        HelpFormatter formatter = new HelpFormatter();

        String input_path = "";
        int k=0;
        int tries=5;
        int chunk_size=1000;
        int max_iterations=40;
        boolean verbose=false;
        boolean print=false;
        List<CentroidCluster<DoublePoint>> clusters;

        try {
            // parse the command line arguments
            CommandLine line = parser.parse(options, args);

            if(line.hasOption("t")){
                tries=Integer.parseInt(line.getOptionValue("t"));
            }

            if(line.hasOption("m")){
                max_iterations=Integer.parseInt(line.getOptionValue("m"));
            }

            if(line.hasOption("c")){
                chunk_size=Integer.parseInt(line.getOptionValue("c"));
            }

            if(line.hasOption("v")){
                verbose=true;
            }

            if(line.hasOption("p")){
                print=true;
            }

            if (!line.hasOption("k")) {
                System.out.println("Number of clusters must be given.");
                formatter.printHelp("mrk-means", options);
                System.exit(-1);
            } else {
                k=Integer.parseInt(line.getOptionValue("k"));
            }

            if (!line.hasOption("i")) {
            System.out.println("An input file must be given.");
            formatter.printHelp("mrk-means", options);
            System.exit(-1);
            } else {
                input_path = line.getOptionValue("i");

                long t0=System.nanoTime();

                if(line.hasOption("standard")){ //standard kmeans
                    System.out.println("Reading the whole dataset....");
                    List<DoublePoint> points = CSVReader.readAllPointsFromFile(input_path);
                    System.out.printf("read %,d ponits.\n",points.size());
                    System.out.printf("Took %,d Milliseconds\n",(System.nanoTime()-t0)/1000000);

                    System.out.println("Using the standard k-means algorithm.");
                    if(line.hasOption("t")){
                        System.out.printf("Will try %d times!\n",tries);
                        MultiKMeans multiKMeans=new MultiKMeans(k,tries);
                        if(verbose)
                            multiKMeans.verbose=true;
                        clusters=multiKMeans.cluster(points);
                    }else{
                        System.out.printf("Max iterations is %d!\n",max_iterations);
                        KMeansClusterer kmeans=new KMeansClusterer(k,max_iterations);
                        clusters=kmeans.cluster(points);
                    }
                }else if(line.hasOption("stream")) { //stream kmeans++
                    System.out.println("Using the stream k-means++ algorithm.");
                    System.out.println("The chunk size is: "+chunk_size);
                    System.out.println("tries per chunk is: "+tries);
                    StreamKMeansPlusPlusClusterer streamKMeansPlusPlus=new StreamKMeansPlusPlusClusterer(input_path);
                    if(verbose)
                        streamKMeansPlusPlus.verbose=true;
                    clusters=streamKMeansPlusPlus.cluster(k,chunk_size,tries);
                }
                else { //kmeans++
                    System.out.println("Reading the whole dataset....");
                    List<DoublePoint> points = CSVReader.readAllPointsFromFile(input_path);
                    System.out.printf("read %,d ponits.\n",points.size());
                    System.out.printf("Took %,d Milliseconds\n",(System.nanoTime()-t0)/1000000);

                    System.out.println("Using the k-means++ algorithm.");
                    if(line.hasOption("t")){
                        System.out.printf("Will try %d times!\n",tries);
                        MultiKMeansPlusPlus multiKMeansPlusPlus=new MultiKMeansPlusPlus(k,tries);
                        if(verbose)
                            multiKMeansPlusPlus.verbose=true;
                        clusters=multiKMeansPlusPlus.cluster(points);
                    }else{
                        System.out.printf("Max iterations is %d!\n",max_iterations);
                        KMeansPlusPlusClusterer kmeans=new KMeansPlusPlusClusterer(k,max_iterations);
                        clusters=kmeans.cluster(points);
                    }
                }

                System.out.printf("Took %,d Milliseconds\n",(System.nanoTime()-t0)/1000000);

                if(print){
                    for(CentroidCluster<DoublePoint> center:clusters)
                        System.out.println(center.getCenter());
                }

                if(verbose){
                    System.out.println("Evaluating clusters...");
                    System.out.printf("ICD is: %g\n",Evaluator.computeICD(clusters));
                    System.out.printf("SSE is: %g\n",Evaluator.computeSSE(clusters,input_path));
                }
            }


        } catch (org.apache.commons.cli.ParseException exp) {
            System.out.println("Unexpected exception:" + exp.getMessage());
            formatter.printHelp("subdigger", options);
            System.exit(-1);
        }



//        List<DoublePoint> points = CSVReader.readAllPointsFromFile("X:\\DataSets\\sets\\s4-15.txt");
//        System.out.printf("points: %,d\n",points.size());
//
//
//        //System.in.read();
//        long t0 = System.nanoTime();
//
//        StreamKMeansPlusPlusClusterer streamKMeansPlusPlus=new StreamKMeansPlusPlusClusterer("X:\\DataSets\\sets\\s4-15.txt");
//        List<CentroidCluster<DoublePoint>> clusters=streamKMeansPlusPlus.cluster(15,50,5);
//        System.out.println("++++++++++++++++++++++++++++++");
//        System.out.println("sse: " + Evaluator.computeSSE(clusters,points));
//        System.out.println("icd: " + Evaluator.computeICD(clusters));
//        System.out.printf("time: %,d\n", System.nanoTime() - t0);
//
//        MultiKMeansPlusPlus multiKMeansPlusPlus=new MultiKMeansPlusPlus(15,5);
//        clusters=multiKMeansPlusPlus.cluster(points);
//        System.out.println("++++++++++++++++++++++++++++++");
//        System.out.println("sse: " + Evaluator.computeSSE(clusters,points));
//        System.out.println("icd: " + Evaluator.computeICD(clusters));
//        System.out.printf("time: %,d\n", System.nanoTime() - t0);
//
//        MultiKMeans multiKMeans=new MultiKMeans(15,5);
//        clusters=multiKMeans.cluster(points);
//        System.out.println("++++++++++++++++++++++++++++++");
//        System.out.println("sse: " + Evaluator.computeSSE(clusters,points));
//        System.out.println("icd: " + Evaluator.computeICD(clusters));
//        System.out.printf("time: %,d\n", System.nanoTime() - t0);


    }
}
