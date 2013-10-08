package org.tmu;

import org.apache.commons.cli.*;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;

import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.tmu.clustering.*;
import org.tmu.util.CSVReader;
import org.tmu.util.GaussianPointGenerator;

import java.io.IOException;
import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 9/20/13
 * Time: 10:01 PM
 * To change this template use File | Settings | File Templates.
 */
public class Main extends Configured implements Tool {

    static String input_path = "";
    static String output_path = "";
    static int k = 0;
    static int tries = 3;
    static int chunk_size = 1000;
    static int max_iterations = 40;
    static boolean verbose = false;
    static boolean print = false;

    public static void main(String[] args) throws IOException {
        GaussianPointGenerator gen=new GaussianPointGenerator(20);
        for(int i=0;i<100000;i++){
            DoublePoint point=gen.nextPoint();
            for(int j=0;j<point.getPoint().length;j++)
                System.out.printf("%f",point.getPoint()[j]);
            System.out.println();
        }
        //parsing command line
        // create the command line parser
        CommandLineParser parser = new BasicParser();
        //System.in.read();

        // create the Options
        Options options = new Options();
        options.addOption("i", "input", true, "the input file name.");
        options.addOption("o", "output", true, "the output path.");
        options.addOption("k", "k", true, "the number of clusters.");
        options.addOption("plus", false, "use kmeans++.");
        options.addOption("standard", false, "use standard kmeans.");
        options.addOption("stream", false, "use stream kmeans++.");
        options.addOption("mapreduce", false, "use Hadoop ;p");
        options.addOption("t", "tries", true, "try x times and return the best.");
        options.addOption("p", "print", false, "print the final centers.");
        options.addOption("m", "max", true, "the max iterations. Also denotes the per chunk iteration for the stream case.");
        options.addOption("c", "chunk", true, "the chunk size.");
        options.addOption("v", "verbose", false, "be verbose.");

        HelpFormatter formatter = new HelpFormatter();

        List<CentroidCluster<DoublePoint>> clusters;

        try {
            // parse the command line arguments
            CommandLine line = parser.parse(options, args);

            if (line.hasOption("t")) {
                tries = Integer.parseInt(line.getOptionValue("t"));
            }

            if (line.hasOption("m")) {
                max_iterations = Integer.parseInt(line.getOptionValue("m"));
            }

            if (line.hasOption("p")) {
                print = true;
            }

            if (line.hasOption("c")) {
                chunk_size = Integer.parseInt(line.getOptionValue("c"));
            }

            if (line.hasOption("v")) {
                verbose = true;
            }

            if (line.hasOption("p")) {
                print = true;
            }

            if (!line.hasOption("k")) {
                System.out.println("Number of clusters must be given.");
                formatter.printHelp("mrk-means", options);
                System.exit(-1);
            } else {
                k = Integer.parseInt(line.getOptionValue("k"));
            }

            if (line.hasOption("o")) {
                output_path = line.getOptionValue("o");
            }

            if (!line.hasOption("i")) {
                System.out.println("An input file must be given.");
                formatter.printHelp("mrk-means", options);
                System.exit(-1);
            } else {
                input_path = line.getOptionValue("i");

                long t0 = System.nanoTime();

                if (line.hasOption("standard")) { //standard kmeans
                    System.out.println("Reading the whole dataset....");
                    List<DoublePoint> points = CSVReader.readAllPointsFromFile(input_path);
                    System.out.printf("read %,d ponits.\n", points.size());
                    System.out.printf("Took %,d Milliseconds\n", (System.nanoTime() - t0) / 1000000);

                    System.out.println("Using the standard k-means algorithm.");
                    if (line.hasOption("t")) {
                        System.out.printf("Will try %d times!\n", tries);
                        MultiKMeans multiKMeans = new MultiKMeans(k, max_iterations, tries);
                        if (verbose)
                            multiKMeans.verbose = true;
                        clusters = multiKMeans.cluster(points);
                    } else {
                        System.out.printf("Max iterations is %d!\n", max_iterations);
                        KMeansClusterer kmeans = new KMeansClusterer(k, max_iterations);
                        clusters = kmeans.cluster(points);
                    }
                } else if (line.hasOption("stream")) { //stream kmeans++
                    if (!line.hasOption("t"))
                        tries = 1;
                    System.out.println("Using the stream k-means++ algorithm.");
                    StreamKMeansPlusPlusClusterer streamKMeansPlusPlus = new StreamKMeansPlusPlusClusterer(input_path);
                    if (verbose)
                        streamKMeansPlusPlus.verbose = true;
                    clusters = streamKMeansPlusPlus.cluster(k, chunk_size, max_iterations, tries);
                } else { //kmeans++
                    System.out.println("Reading the whole dataset....");
                    List<DoublePoint> points = CSVReader.readAllPointsFromFile(input_path);
                    System.out.printf("read %,d ponits.\n", points.size());
                    System.out.printf("Took %,d Milliseconds\n", (System.nanoTime() - t0) / 1000000);

                    System.out.println("Using the k-means++ algorithm.");
                    if (line.hasOption("t")) {
                        System.out.printf("Will try %d times!\n", tries);
                        MultiKMeansPlusPlus multiKMeansPlusPlus = new MultiKMeansPlusPlus(k, max_iterations, tries);
                        if (verbose)
                            multiKMeansPlusPlus.verbose = true;
                        clusters = multiKMeansPlusPlus.cluster(points);
                    } else {
                        System.out.printf("Max iterations is %d!\n", max_iterations);
                        KMeansPlusPlusClusterer kmeans = new KMeansPlusPlusClusterer(k, max_iterations);
                        clusters = kmeans.cluster(points);
                    }
                }

                System.out.printf("Took %,d Milliseconds\n", (System.nanoTime() - t0) / 1000000);

                if (print) {
                    for (CentroidCluster<DoublePoint> center : clusters)
                        System.out.println(center.getCenter());
                }

                if (verbose) {
                    System.out.println("Evaluating clusters...");
                    System.out.printf("ICD is: %g\n", Evaluator.computeICD(clusters));
                    System.out.printf("SSE is: %g\n", Evaluator.computeSSE(clusters, input_path));
                }
            }


        } catch (org.apache.commons.cli.ParseException exp) {
            System.out.println("Unexpected exception:" + exp.getMessage());
            formatter.printHelp("subdigger", options);
            System.exit(-1);
        }

    }

    @Override
    public int run(String[] strings) throws Exception {
        Configuration conf = getConf();
        Job job = new Job(conf, "MRSUB-" + new Path(input_path).getName() + "-" + k);
        final Path inDir = new Path(input_path);
        if (output_path.isEmpty()) {
            System.out.println("Must define an output path!");
            System.exit(0);
        }
        final Path outDir = new Path(output_path);

        job.setJarByClass(Main.class);
//        job.setMapperClass(SubMapper.class);
//        job.setCombinerClass(SubCombiner.class);
//        job.setReducerClass(LongSumReducer.class);
//        job.setOutputKeyClass(Text.class);
//        job.setOutputValueClass(LongWritable.class);

        FileInputFormat.addInputPath(job, inDir);
        job.setInputFormatClass(org.apache.hadoop.mapreduce.lib.input.TextInputFormat.class);
        FileOutputFormat.setOutputPath(job, outDir);
        job.setNumReduceTasks(1);


        int result = job.waitForCompletion(true) ? 0 : 1;
        return result;

    }
}
