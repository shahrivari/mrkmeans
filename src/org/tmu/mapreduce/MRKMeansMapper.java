package org.tmu.mapreduce;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.tmu.util.CSVReader;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 9/30/13
 * Time: 11:53 AM
 * To change this template use File | Settings | File Templates.
 */
public class MRKMeansMapper extends Mapper<LongWritable, Text, IntWritable , PointWritable> {
    List<DoublePoint> chunk=new ArrayList<DoublePoint>();
    int point_size=-1;
    int k=0;
    int chunk_size=1000;
    int iterations=1;

    @Override
    protected void setup(Context context) throws IOException {
        k=context.getConfiguration().getInt("ClusterCount",2);
        chunk_size=context.getConfiguration().getInt("ChunkSize",1000);
        iterations=context.getConfiguration().getInt("Iterations",1);
        chunk.clear();
    }

    @Override
    protected void map(LongWritable key,Text val,Context context) throws IOException, InterruptedException {
        String line=val.toString();

        if(line.isEmpty()){
            context.getCounter(Counters.EMPTY_LINE).increment(1);
            return;
        }
        DoublePoint point=null;
        try {
            point=CSVReader.ReadFromString(line);
        }catch (NumberFormatException exp){
            context.getCounter(Counters.BAD_POINT).increment(1);
            return;
        }

        if(point_size==-1)
            point_size=point.getPoint().length;

        if(point.getPoint().length!=point_size){
            context.getCounter(Counters.BAD_POINT).increment(1);
            return;
        }
        context.getCounter(Counters.GOOD_POINT).increment(1);

        chunk.add(point);
        if(chunk.size()>=chunk_size){
            KMeansPlusPlusClusterer<DoublePoint> kmeanspp=new KMeansPlusPlusClusterer<DoublePoint>(k,iterations,new EuclideanDistance());
            List<CentroidCluster<DoublePoint>> clusters=kmeanspp.cluster(chunk);
            chunk.clear();
            for(CentroidCluster<DoublePoint> cluster: clusters)
                context.write(new IntWritable(1), new PointWritable(cluster.getCenter().getPoint()));
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        if(chunk.size()>=k){
            KMeansPlusPlusClusterer<DoublePoint> kmeanspp=new KMeansPlusPlusClusterer<DoublePoint>(k,iterations,new EuclideanDistance());
            List<CentroidCluster<DoublePoint>> clusters=kmeanspp.cluster(chunk);
            chunk.clear();
            for(CentroidCluster<DoublePoint> cluster: clusters)
                context.write(new IntWritable(1), new PointWritable(cluster.getCenter().getPoint()));
        }
    }
}
