package org.tmu.util;

import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.random.*;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 10/8/13
 * Time: 4:06 PM
 * To change this template use File | Settings | File Templates.
 */
public class GaussianPointGenerator {
    public double[][] means;
    int seed=12345;
    int d=15;
    double len=50;
    double std =1.0;
    double[] std_arr;
    double sse=0;
    EuclideanDistance distance=new EuclideanDistance();

    RandomDataGenerator random_data =new RandomDataGenerator();
    RandomGenerator rg = new JDKRandomGenerator();
    GaussianRandomGenerator rawGenerator = new GaussianRandomGenerator(rg);

    public GaussianPointGenerator(int k,int d,int max_val){
        this.d=d;
        this.len=max_val;
        means=new double[k][d];
        //random_data.reSeed(seed);

        for(int i=0;i<k;i++){
//            boolean pass=true;
//            do{
//                means[i]=randomPoint(d,len);
//                for(int j=0;j<i;j++)
//                    if(distance.compute(means[j],means[i])<2*std)
//                        pass=false;
//            }while (!pass);
            means[i]=randomPoint(d,len);
        }

        std_arr=new double[d];
        for(int i=0;i<std_arr.length;i++)
            std_arr[i]=std;
    }


    private double[] randomPoint(int d,double max){
        double[] res=new double[d];
        for(int i=0;i<d;i++)
            res[i]=(random_data.nextUniform(0.0,1.0))*max;
        return res;
    }

    private double[] randomPoint(double[] mean){
        UncorrelatedRandomVectorGenerator randvec=new UncorrelatedRandomVectorGenerator(mean,std_arr,rawGenerator);
        return  randvec.nextVector();
    }

    public DoublePoint nextPoint(){
        double[] mean;
        if(means.length==1)
            mean=means[0];
        else
            mean=means[random_data.nextInt(0,means.length-1)];

        double [] point=randomPoint(mean);
        double dis=distance.compute(mean,point);
        sse+=dis*dis;
        return new DoublePoint(point);
    }
}
