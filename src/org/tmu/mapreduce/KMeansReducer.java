package org.tmu.mapreduce;

import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.tmu.util.CSVReader;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
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
public class KMeansReducer extends Reducer<IntWritable, PointWritable,Text,Text> {

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
    protected void reduce(IntWritable key, Iterable<PointWritable> values, Context context) throws IOException, InterruptedException {
        //sse
        if(key.get()==-1){
            double sse=0;
            for(PointWritable p:values)
                sse+=p.point.getPoint()[0];
            context.write(new Text("SSE:"),new Text(Double.toString(sse)));
            return;
        }

        DoublePoint center=new DoublePoint(new double[point_size]);
        for(int i=0;i<point_size;i++)
            center.getPoint()[i]=0.0;

        long count=0;
        for(PointWritable point:values)
            for(int i=0;i<point_size;i++){
                center.getPoint()[i]+=point.point.getPoint()[i];
                count++;
            }

        for(int i=0;i<point_size;i++)
            center.getPoint()[i]=center.getPoint()[i]/count;
        context.write(new Text("Center: "), new Text(Arrays.toString(center.getPoint())));
    }

}
