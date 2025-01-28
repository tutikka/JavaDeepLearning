package com.tt.javadeeplearning.metrics;

public class Metrics {

    public static double accuracy(double[] truth, int[] predicted) {
        int correct = 0;
        for (int i = 0; i < truth.length; i++) {
            int t = (int) truth[i];
            int p = predicted[i];
            if (p == t) {
                correct++;
            }
        }
        return ((double) correct / (double) truth.length);
    }

    public static double binaryPrecision(double[] truth, int[] predicted) {
        int truePositives = 0;
        int falsePositives = 0;
        for (int i = 0; i < truth.length; i++) {
            int t = (int) truth[i];
            int p = predicted[i];
            if (p == 1) {
                if (t == 1) {
                    truePositives++;
                } else {
                    falsePositives++;
                }
            }
        }
        if (truePositives + falsePositives == 0) {
            return (0.0);
        }
        return ((double) truePositives / (truePositives + falsePositives));
    }

    public static double macroPrecision(double[] truth, int[] predicted, int numClasses) {
        double[] classPrecisions = classWisePrecision(truth, predicted, numClasses);
        double sum = 0.0;
        for (double precision : classPrecisions) {
            sum += precision;
        }
        return (sum / numClasses);
    }

    public static double weightedPrecision(double[] truth, int[] predicted, int numClasses) {
        double[] classPrecisions = classWisePrecision(truth, predicted, numClasses);
        int[] support = new int[numClasses];
        for (int i = 0; i < truth.length; i++) {
            support[(int) truth[i]]++;
        }
        double weightedSum = 0.0;
        int totalSamples = truth.length;
        for (int c = 0; c < numClasses; c++) {
            weightedSum += classPrecisions[c] * support[c];
        }
        return (weightedSum / totalSamples);
    }

    private static double[] classWisePrecision(double[] truth, int[] predicted, int numClasses) {
        int[] truePositives = new int[numClasses];
        int[] falsePositives = new int[numClasses];
        for (int i = 0; i < truth.length; i++) {
            double t = truth[i];
            double p = predicted[i];
            for (int c = 0; c < numClasses; c++) {
                if (p == c) {
                    if (t == c) {
                        truePositives[c]++;
                    } else {
                        falsePositives[c]++;
                    }
                }
            }
        }
        double[] precisions = new double[numClasses];
        for (int c = 0; c < numClasses; c++) {
            if (truePositives[c] + falsePositives[c] > 0) {
                precisions[c] = (double) truePositives[c] / (truePositives[c] + falsePositives[c]);
            } else {
                precisions[c] = 0.0;
            }
        }
        return precisions;
    }

}
