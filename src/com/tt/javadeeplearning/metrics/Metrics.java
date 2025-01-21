package com.tt.javadeeplearning.metrics;

public class Metrics {

    public static double accuracy(double[] truth, double[] predicted) {
        int correct = 0;
        for (int i = 0; i < truth.length; i++) {
            double t = truth[i];
            double p = predicted[i];
            if (p == t) {
                correct++;
            }
        }
        return ((double) correct / (double) truth.length);
    }

}
