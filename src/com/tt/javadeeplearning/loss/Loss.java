package com.tt.javadeeplearning.loss;

public interface Loss {

    double loss(double truth, double[] predicted);

    double[] gradient(double truth, double[] predicted);

}
