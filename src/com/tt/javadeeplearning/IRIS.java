package com.tt.javadeeplearning;

import com.tt.javadeeplearning.activation.TanH;
import com.tt.javadeeplearning.initialization.Xavier;
import com.tt.javadeeplearning.layer.Activation;
import com.tt.javadeeplearning.layer.Dense;
import com.tt.javadeeplearning.layer.Softmax;
import com.tt.javadeeplearning.loss.CategoricalCrossEntropy;
import com.tt.javadeeplearning.network.Network;
import com.tt.javadeeplearning.postprocess.PostProcess;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class IRIS {

    public static int labelFromString(String label) {
        switch (label) {
            case "Iris-setosa": return (0);
            case "Iris-versicolor": return (1);
            case "Iris-virginica":return (2);
        }
        return (-1);
    }

    public static String labelFromInt(int i) {
        switch (i) {
            case 0: return ("Iris-setosa");
            case 1: return ("Iris-versicolor");
            case 2: return ("Iris-virginica");
        }
        return (null);
    }

    public static double[] labels(String path) {
        try {
            List<Double> l = new ArrayList<>();
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                String label = line.split(",")[4];
                l.add((double) labelFromString(label));
            }
            return (l.stream().mapToDouble(d -> d).toArray());
        } catch (Exception e) {
            return (null);
        }
    }

    public static double[][] features(String path) {
        try {
            List<double[]> l = new ArrayList<>();
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                double sepalLength = Double.parseDouble(line.split(",")[0]);
                double sepalWidth = Double.parseDouble(line.split(",")[1]) ;
                double petalLength = Double.parseDouble(line.split(",")[2]);
                double petalWidth = Double.parseDouble(line.split(",")[3]);
                l.add(new double[]{sepalLength, sepalWidth, petalLength, petalWidth});
            }
            double[][] d = new double[l.size()][];
            for (int i = 0; i < d.length; i++) {
                d[i] = l.get(i);
            }
            return (d);
        } catch (Exception e) {
            return (null);
        }
    }

    public static void main(String[] args) {

        double[][] features = features("data/iris/iris.data");
        double[] labels = labels("data/iris/iris.data");

        int split = (int) (labels.length * 0.8);

        double[][] trainX = Arrays.stream(features, 0, split).toArray(double[][]::new);
        double[] trainY = Arrays.stream(labels, 0, split).toArray();

        Network network = new Network();
        network.addLayer(new Dense(4, 8, new Xavier()));
        network.addLayer(new Activation(new TanH()));
        network.addLayer(new Dense(8, 3, new Xavier()));
        network.addLayer(new Softmax());

        network.train(trainX, trainY, new CategoricalCrossEntropy(), 1000, 0.01);

        network.save(new File("iris.ser"));

        network = Network.load(new File("iris.ser"));
        System.out.println(network);

        double[][] testX = Arrays.stream(features, split, labels.length).toArray(double[][]::new);
        double[] testY = Arrays.stream(labels, split, labels.length).toArray();

        int a = 0;
        for (int i = 0; i < testY.length; i++) {
            double t = testY[i];
            double p = PostProcess.argmax(network.predict(testX[i]));
            if (p == t) {
                a++;
            }
        }
        System.out.print(String.format("accuracy based on test set of %d items is %s ", testY.length, (double) a / (double) testY.length));
    }

}
