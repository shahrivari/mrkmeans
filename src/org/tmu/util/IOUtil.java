package org.tmu.util;

import org.apache.commons.math3.ml.clustering.DoublePoint;

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

    public static String VectorToCompactString(double[] point)
    {
        StringBuilder builder=new StringBuilder();
        //DecimalFormat threeDec = new DecimalFormat("0.000");
        for(int i=0;i<point.length;i++){
            //builder.append(threeDec.format(elements[i])).append(",");
            //some dirty code to convert double to string with 4 precision
            String s=Double.toString(point[i]);
            int dot_place=s.indexOf('.');
            if(dot_place+5<s.length())
                s=s.substring(0,dot_place+5);
            builder.append(s).append(" ");
        }
        builder.deleteCharAt(builder.length()-1);
        return builder.toString();
    }

    public static String PointToCompactString(DoublePoint point){
        return VectorToCompactString(point.getPoint());
    }



}
