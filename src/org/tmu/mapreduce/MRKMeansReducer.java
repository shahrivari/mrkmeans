package org.tmu.mapreduce;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapreduce.Reducer;
import org.tmu.clustering.MultiKMeansPlusPlus;
import org.tmu.util.CSVReader;
import org.tmu.util.IOUtil;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 9/30/13
 * Time: 11:53 AM
 * To change this template use File | Settings | File Templates.
 */
public class MRKMeansReducer extends Reducer<IntWritable, PointWritable,Text,Text> {

    List<DoublePoint> centers=new ArrayList<DoublePoint>();
    int point_size=-1;
    int k=0;

    @Override
    protected void setup(Context context) throws IOException {
        k=context.getConfiguration().getInt("ClusterCount",2);
    }


    @Override
    protected void reduce(IntWritable key, Iterable<PointWritable> values, Context context) throws IOException, InterruptedException {
        //The intermediate centers
        if(key.get()==1){
            double sse=0;

            Path out=new Path(context.getConfiguration().get("mapred.output.dir")+"/centers.txt");
            FileSystem fs=FileSystem.get(context.getConfiguration());
            FSDataOutputStream stream=fs.create(out,true);
            BufferedWriter writer = new BufferedWriter( new OutputStreamWriter( stream, "UTF-8" ) );

            for(PointWritable p:values){
                centers.add(p.point);
                context.getCounter(Counters.INTERMEDIATE_CENTERS).increment(1);
                writer.write(IOUtil.PointToString(p.point)+"\n");
            }

            writer.close();

            MultiKMeansPlusPlus last_kmeanspp = new MultiKMeansPlusPlus(k, (int) Math.log(centers.size()) * 2, 3);
            last_kmeanspp.verbose=true;
            List<CentroidCluster<DoublePoint>> clusters=last_kmeanspp.cluster(centers);
            for(CentroidCluster<DoublePoint> cluster: clusters)
                context.write(new Text(IOUtil.PointToString(cluster.getCenter().getPoint())),new Text(""));

            return;
        }
    }

}
