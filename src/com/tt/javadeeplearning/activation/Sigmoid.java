package com.tt.javadeeplearning.activation;

public class Sigmoid implements Activation {

    @Override
    public double activate(double d) {
        return (1 / (1 + Math.exp(-d)));
    }

    @Override
    public double gradient(double d) {
        double result = activate(d);
        return (result * (1 - result));
    }
}
