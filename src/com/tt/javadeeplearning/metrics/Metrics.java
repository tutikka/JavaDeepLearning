package com.tt.javadeeplearning.metrics;

public class Metrics {

    public static double accuracy(double[] truth, double[] predicted) {
        int numCorrect = 0;
        for (int i = 0; i < truth.length; i++) {
            if (predicted[i] == truth[i]) {
                numCorrect++;
            }
        }
        double a = (double) numCorrect / (double) truth.length;
        return (a);
    }

}
