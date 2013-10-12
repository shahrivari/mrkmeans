package org.tmu.util;

import org.apache.commons.math3.ml.clustering.DoublePoint;

import java.util.Arrays;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 10/12/13
 * Time: 5:44 PM
 * To change this template use File | Settings | File Templates.
 */
public class IOUtil {

    public static String PointToString(double [] point){
        StringBuilder builder=new StringBuilder();
        for(double d:point){
            builder.append(d);
            builder.append(',');
        }
        String str=builder.toString();
        if(str.length()>0)
            return str.substring(0,str.length()-1);
        else
            return str;
    }

    public static String PointToString(DoublePoint point){
        return PointToString(point.getPoint());
    }


}
