package org.tmu.clustering;

import org.apache.commons.math3.ml.distance.DistanceMeasure;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 9/21/13
 * Time: 6:39 PM
 * To change this template use File | Settings | File Templates.
 */
public class SquaredEuclideanDistance implements DistanceMeasure {
    @Override
    public double compute(double[] a, double[] b) {
        double distance = 0.0;
        if (b.length != a.length)
            throw new IllegalArgumentException("Target point's size is not equal!");
        for (int i = 0; i < a.length; i++)
            distance += (a[i] - b[i]) * (a[i] - b[i]);
        return distance;

    }
}
