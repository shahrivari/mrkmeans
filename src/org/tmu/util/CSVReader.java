package org.tmu.util;


import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Created with IntelliJ IDEA.
 * User: Saeed
 * Date: 10/13/12
 * Time: 11:37 PM
 * To change this template use File | Settings | File Templates.
 */
public class CSVReader {
    BufferedReader reader;
    FileReader fileReader;

    public void close() throws IOException {
        reader.close();
        fileReader.close();
    }

    @Override
    protected void finalize() throws IOException {
        close();
    }

    public CSVReader(String file_name) throws FileNotFoundException {
        fileReader = new FileReader(file_name);
        reader = new BufferedReader(fileReader);
    }

    private AtomicLong lineNumber=new AtomicLong(0);

    private String readNextNonEmptyLine() throws IOException {
        String line = "";
        do {
            line = reader.readLine();
            lineNumber.getAndIncrement();
            if (line == null) return null;
            line = line.trim();
        } while (line.isEmpty());
        return line;
    }

    public double[] ReadNextDoubleVector() throws IOException {
        while (true){
        String line = readNextNonEmptyLine();
        if (line == null) return null;

        //String[] tokens = line.split("\\s*(;|,|\\s)\\s*");
        //String[] tokens = line.split("\\s|,|;");
        String[] tokens= StringUtils.split(line," ,;\t");

        double[] point = new double[tokens.length];

        for (int i = 0; i < point.length; i++)
            try {
                point[i] = Double.parseDouble(tokens[i]);
            } catch (NumberFormatException e) {
                e.printStackTrace();
                System.out.println("Error reading line: " + lineNumber);
                continue;
            }

        return point;
        }
    }

    public static List<double[]> readWholeFile(String file_name) throws IOException {
        CSVReader csvReader = new CSVReader(file_name);
        List<double[]> result = new ArrayList<double[]>(1024);
        do {
            double[] vec = csvReader.ReadNextDoubleVector();
            if (vec == null)
                break;
            result.add(vec);
        } while (true);
        return result;
    }


}
