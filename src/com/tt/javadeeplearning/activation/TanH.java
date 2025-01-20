package com.tt.javadeeplearning.activation;

import java.io.Serializable;

public class TanH implements Activation, Serializable {

    @Override
    public double activate(double d) {
        return (Math.tanh(d));
    }

    @Override
    public double gradient(double d) {
        return ((1 - Math.pow(d, 2)));
    }

    public String toString() {
        return ("TanH");
    }

}
