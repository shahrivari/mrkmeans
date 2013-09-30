package org.tmu.mapreduce;

import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 9/30/13
 * Time: 1:32 PM
 * To change this template use File | Settings | File Templates.
 */
public class PointWritable implements WritableComparable {
    DoublePoint point;

    PointWritable(DoublePoint p) {
        this.point = point;
    }

    private PointWritable() {
        point = null;
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        double[] arr = point.getPoint();
        dataOutput.writeInt(arr.length);
        for (int i = 0; i < arr.length; i++)
            dataOutput.writeDouble(arr[i]);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        int size = dataInput.readInt();
        double[] arr = new double[size];
        for (int i = 0; i < arr.length; i++)
            arr[i] = dataInput.readDouble();
        point = new DoublePoint(arr);
    }

    @Override
    public int compareTo(Object o) {
        if (o instanceof PointWritable) {
            PointWritable pointWritable = (PointWritable) o;
            double[] arr1 = pointWritable.point.getPoint();
            double[] arr2 = point.getPoint();
            if (arr1.length != arr2.length)
                throw new IllegalArgumentException("Point sizes are not the same!");
            for (int i = 0; i < arr1.length; i++)
                if (arr1[i] < arr2[i])
                    return -1;
                else if (arr1[i] > arr2[i])
                    return 1;
            return 0;
        }
        throw new IllegalArgumentException("argumet is not PointWritable!");
    }
}
