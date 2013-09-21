package org.tmu.clustering;

import org.apache.commons.math3.ml.clustering.Clusterable;
import org.tmu.util.CSVReader;

import java.io.IOException;
import java.util.*;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 10/9/12
 * Time: 7:52 PM
 * To change this template use File | Settings | File Templates.
 */
public class WeightedPoint implements Clusterable {
    public double[] elements=null;
    double weight=1.0;

    @Override
    public double[] getPoint() {
        return elements;
    }

    public WeightedPoint()
    {
        elements=new double[0];
    }

    public WeightedPoint(int size)
    {
        elements=new double[size];
        for(int i=0;i<size;i++)
            elements[i]=0.0;
    }

    public WeightedPoint(double[] point)
    {
        elements=new double[point.length];
        System.arraycopy(point, 0, elements, 0, point.length);
    }

    public WeightedPoint(WeightedPoint point)
    {
        this(point.elements);
        weight=point.weight;
    }

    public double getWeight() {
        return weight;
    }


    public double getElement(int index) {
        return elements[index];
    }

    public void setElement(int index, double value)
    {
        elements[index]=value;
    }

    public int size()
    {
        return elements.length;
    }

    @Override
    public String toString()
    {
        return Arrays.toString(elements);
    }

    public double distanceFrom(WeightedPoint point) {
        double distance=0.0;
        if(size()!=point.size())
            throw new IllegalArgumentException("Target point's size is not equal!");
        for(int i=0;i<size();i++)
            distance+=(elements[i]-point.getElement(i))*(elements[i]-point.getElement(i));
        return distance*weight;
    }

    public WeightedPoint centroidOf(List<WeightedPoint> points) {
        if(points.size()==0)
            throw new IllegalArgumentException("There is no points!");

        WeightedPoint result=new WeightedPoint(points.get(0).size());

        for(int x=0;x<points.size();x++ )
        {
            WeightedPoint p=points.get(x);
            if(p.size()!=result.size())
                throw new IllegalArgumentException("There is a point with mismatched size: "+ p.toString());
            for (int i=0;i<p.size();i++)
                result.setElement(i,result.getElement(i)+p.getElement(i));
        }

        for (int i=0;i<result.size();i++)
            result.setElement(i,result.getElement(i)/points.size());

        return result;
    }

    public WeightedPoint findNearest(List<WeightedPoint> points){
        WeightedPoint result=null;
        double  min_dis=Double.MAX_VALUE;
        for(int x=0;x<points.size();x++ ){
            WeightedPoint p=points.get(x);
            if(distanceFrom(p)<min_dis){
                min_dis=distanceFrom(p);
                result=p;
            }
        }
        return result;
    }

    static public List<WeightedPoint> readAllPointsFromFile(String path) throws IOException {
        CSVReader reader=new CSVReader(path);
        List<WeightedPoint> result=new ArrayList<WeightedPoint>();
        do{
            double [] vec=reader.ReadNextDoubleVector();
            if(vec==null)
                break;
            result.add(new WeightedPoint(vec));

        }while (true);
        return result;
    }
}
