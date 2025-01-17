package com.tt.javadeeplearning.activation;

public class ReLu implements Activation {

    @Override
    public double activate(double d) {
        return (Math.max(0, d));
    }

    @Override
    public double gradient(double d) {
        return (d > 0 ? 1 : 0);
    }

}
