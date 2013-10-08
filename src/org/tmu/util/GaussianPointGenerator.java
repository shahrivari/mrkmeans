package org.tmu.util;

import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.random.*;

import java.util.Iterator;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 10/8/13
 * Time: 4:06 PM
 * To change this template use File | Settings | File Templates.
 */
public class GaussianPointGenerator {
    double[][] means;
    int seed=1234;
    int d=15;
    double len=50;
    double radius=1.0;
    double sse=0;
    EuclideanDistance distance=new EuclideanDistance();

    RandomDataGenerator random=new RandomDataGenerator();

    public GaussianPointGenerator(int k){
        means=new double[k][d];
        random.reSeed(seed);
        for(int i=0;i<k;i++){
            boolean pass=true;
            do{
                means[i]=randomPoint(d,len);
                for(int j=0;j<i;j++)
                    if(distance.compute(means[j],means[i])<3*radius)
                        pass=false;
            }while (!pass);
        }
    }


    private double[] randomPoint(int d,double max){
        double[] res=new double[d];
        for(int i=0;i<d;i++)
            res[i]=(random.nextUniform(0.0,1.0))*max;
        return res;
    }

    private double[] randomPoint(double[] mean , double radius){
        double[] res=new double[mean.length];
        for(int i=0;i<res.length;i++)
            res[i]=random.nextUniform(-radius,radius)+mean[i];
        return res;
    }

    public DoublePoint nextPoint(){
        double[] mean;
        if(means.length==1)
            mean=means[0];
        else
            mean=means[random.nextInt(0,means.length-1)];

        double [] point=randomPoint(mean,radius);
        sse+=distance.compute(mean,point);
        return new DoublePoint(point);
    }
}
