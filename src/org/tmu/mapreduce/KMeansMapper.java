package org.tmu.mapreduce;

import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
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
public class KMeansMapper extends Mapper<LongWritable, Text, IntWritable, PointWritable> {
    List<DoublePoint> centers=null;
    int point_size=-1;
    EuclideanDistance distance=new EuclideanDistance();

    @Override
    protected void setup(Context context) throws IOException {
        Path centers_path= new Path(context.getConfiguration().get("CentersPath"));
        FileSystem fs = FileSystem.get(context.getConfiguration());
        BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(centers_path)));

        centers=new ArrayList<DoublePoint>();
        try {
            String line=null;
            do{
                line=br.readLine();
                while (line.isEmpty())
                    continue;
                if(line==null)
                    break;
                DoublePoint point= CSVReader.ReadFromString(line);
                if(point_size==-1)
                    point_size=point.getPoint().length;
                if(point_size!=point.getPoint().length)
                    throw new IllegalStateException("Centers in the file do not have the same size!");
                centers.add(point);
            }while (line!=null);

        } finally {
            // you should close out the BufferedReader
            br.close();
        }
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

        if(point.getPoint().length!=point_size){
            context.getCounter(Counters.BAD_POINT).increment(1);
            return;
        }

        double min_dis=Double.MAX_VALUE;
        int nearest=0;
        for(int i=0;i<centers.size();i++)
            if(distance.compute(centers.get(i).getPoint(),point.getPoint())<min_dis){
                min_dis=distance.compute(centers.get(i).getPoint(),point.getPoint());
                nearest=i;
            }
        DoublePoint sse=new DoublePoint(new double[point_size]);
        sse.getPoint()[0]=min_dis;
        context.write(new IntWritable(-1),new PointWritable(sse));
        context.write(new IntWritable(nearest),new PointWritable(point));
        context.getCounter(Counters.GOOD_POINT).increment(1);
    }

}
