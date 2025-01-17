package com.tt.javadeeplearning.activation;

public class TanH implements Activation {

    @Override
    public double activate(double d) {
        return (Math.tanh(d));
    }

    @Override
    public double gradient(double d) {
        return ((1 - Math.pow(d, 2)));
    }
}
