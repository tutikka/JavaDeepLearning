package com.tt.javadeeplearning.layer;

import java.io.Serializable;

public abstract class Layer implements Serializable {

    double[] input;

    double[] output;

    public abstract double[] forwardSingle(double[] input);

    public abstract double[] backwardSingle(double[] gradient, double learningRate);

}
