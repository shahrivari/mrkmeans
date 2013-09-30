package org.tmu.mapreduce;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 9/30/13
 * Time: 11:53 AM
 * To change this template use File | Settings | File Templates.
 */
public class KMeansMapper extends Mapper<LongWritable, Text, Text, LongWritable> {
}
