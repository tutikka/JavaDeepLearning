package com.tt.javadeeplearning.loss;

public class BinaryCrossEntropy implements Loss {

    @Override
    public double loss(double truth, double[] predicted) {
        double y = predicted[0];
        y = Math.max(1e-15, Math.min(1 - 1e-15, y));
        return (-(truth * Math.log(y) + (1 - truth) * Math.log(1 - y)));
    }

    @Override
    public double[] gradient(double truth, double[] predicted) {
        double y = predicted[0];
        y = Math.max(1e-15, Math.min(1 - 1e-15, y));
        double gradient = (y - truth) / (y * (1 - y));
        return (new double[] { gradient });
    }
}
