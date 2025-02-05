package com.tt.javadeeplearning.network;

import com.tt.javadeeplearning.layer.Dropout;
import com.tt.javadeeplearning.layer.Layer;
import com.tt.javadeeplearning.loss.Loss;
import com.tt.javadeeplearning.loss.RootMeanError;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Network implements Serializable {

    private List<Layer> layers;

    public Network() {
        layers = new ArrayList<>();
    }

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public void save(File file) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(file, false));
            out.writeObject(this);
            out.close();
        } catch (Exception e) {
        }
    }

    public static Network load(File file) {
        try {
            ObjectInputStream in = new ObjectInputStream(new FileInputStream(file));
            Network network = (Network) in.readObject();
            in.close();
            return (network);
        } catch (Exception e) {
            return (null);
        }
    }

    public void setTrain(boolean train) {
        for (Layer layer : layers) {
            if (layer instanceof Dropout) {
                ((Dropout) layer).setTrain(train);
            }
        }
    }

    /**
     * Predict labels based on features.
     *
     * @param input The features
     * @return The predicted labels
     */
    public double[] predict(double[] input) {
        double[] output = input;
        for (Layer layer : layers) {
            output = layer.forwardSingle(output);
        }
        return (output);
    }

    /**
     * Train the network. Uses RootMeanError as the loss function, 1000 epochs and 0.01 learning rate.
     *
     * Using the XOR function as an example, the features would be the combinations of 0 and 1, while the labels would be the results of each combination:
     *
     * <pre>
     *     {@code
     *     double[][] features = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
     *     double[] trainY = new double[]{0, 1, 1, 0};
     *     }
     * </pre>
     *
     * @param features A vector of combinations of features
     * @param labels A vector of labels for each respective combination of features
     */
    public void train(double[][] features, double[] labels) {
        train(features, labels, new RootMeanError(), 1000, 0.01);
    }

    /**
     * Train the network.
     *
     * Using the XOR function as an example, the features would be the combinations of 0 and 1, while the labels would be the results of each combination:
     *
     * <pre>
     *     {@code
     *     double[][] features = new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
     *     double[] trainY = new double[]{0, 1, 1, 0};
     *     }
     * </pre>
     *
     * @param features A vector of combinations of features
     * @param labels A vector of labels for each respective combination of features
     * @param loss The loss function
     * @param epochs The number of epochs
     * @param learningRate The learning rate
     */
    public void train(double[][] features, double[] labels, Loss loss, int epochs, double learningRate) {
        for (int e = 0; e < epochs; e++) {
            double averageLoss = 0;
            for (int b = 0; b < features.length; b++) {
                double[] x = features[b];
                double y = labels[b];
                double[] output = predict(x);
                double l = loss.loss(y, output);
                averageLoss += l;
                double[] g = loss.gradient(y, output);
                for (int i = layers.size() - 1; i >= 0; i--) {
                    Layer layer = layers.get(i);
                    g = layer.backwardSingle(g, learningRate);
                }
            }
            averageLoss /= features.length;
            System.out.println(String.format("epoch %d/%d | average loss %f", (e + 1), epochs, averageLoss));
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (Layer layer : layers) {
            sb.append(layer);
            sb.append("\n");
        }
        return (sb.toString());
    }

}
