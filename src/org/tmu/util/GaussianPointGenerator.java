package org.tmu.util;

import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.random.*;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 10/8/13
 * Time: 4:06 PM
 * To change this template use File | Settings | File Templates.
 */
public class GaussianPointGenerator {
    public double[][] means;
    int d=15;
    double len=50;
    double std =1.0;
    double[] std_arr;
    double sse=0;
    EuclideanDistance distance=new EuclideanDistance();

    RandomDataGenerator random_data =new RandomDataGenerator();
    RandomGenerator rg = new JDKRandomGenerator();
    GaussianRandomGenerator rawGenerator = new GaussianRandomGenerator(rg);

    public GaussianPointGenerator(int k,int d,int max_val,double std){
        this.d=d;
        this.len=max_val;
        means=new double[k][d];
        for(int i=0;i<k;i++){
            means[i]=randomPoint(d,len);
        }

        this.std=std;
        std_arr=new double[d];
        for(int i=0;i<std_arr.length;i++)
            std_arr[i]=std;
    }

    public void reseed(long seed){
        random_data.reSeed(seed);
        rg.setSeed(seed);
        means=new double[means.length][d];
        for(int i=0;i<means.length;i++){
            means[i]=randomPoint(d,len);
        }

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

    synchronized public DoublePoint nextPoint(){
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

    public static void parallelGenerate(String out_path, int k, int d, long n, int max_val,double std, long seed) throws IOException, InterruptedException {
        final GaussianPointGenerator generator=new GaussianPointGenerator(k,d,max_val,std);
        if(seed!=0)
            generator.reseed(seed);
        final AtomicLong remaining=new AtomicLong(n);
        final ReentrantLock lock=new ReentrantLock();
        final BufferedWriter writer=new BufferedWriter(new FileWriter(out_path),4096*1024);

        Thread[] threads=new Thread[Math.max(Runtime.getRuntime().availableProcessors()/2,4)];
        for(int i=0;i<threads.length;i++){
            threads[i]=new Thread(
                    new Runnable() {
                        @Override
                        public void run() {
                            StringBuilder builder=new StringBuilder(32*1024);
                            while (remaining.get()>0){
                                remaining.decrementAndGet();
                                DoublePoint point=generator.nextPoint();
                                builder.append(IOUtil.PointToCompactString(point));
                                builder.append('\n');
                                if(builder.length()>32*1024){
                                    lock.lock();
                                    try {
                                        writer.write(builder.toString());
                                        remaining.get();
                                    } catch (Exception e) {
                                        e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
                                        lock.unlock();
                                        System.exit(0);
                                    }
                                    builder.setLength(0);
                                    lock.unlock();
                                }
                            }

                            if (builder.length()>0){
                                lock.lock();
                                try {
                                    writer.write(builder.toString());
                                } catch (IOException e) {
                                    e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
                                    lock.unlock();
                                    System.exit(0);
                                }
                                builder.setLength(0);
                                lock.unlock();
                            }
                        }
                    }
            );
        }

        for (int i=0;i<threads.length;i++)
            threads[i].start();

        for (int i=0;i<threads.length;i++)
            threads[i].join();
        writer.close();
    }
}
