package com.tt.javadeeplearning.activation;

import java.io.Serializable;

public class Sigmoid implements Activation, Serializable {

    @Override
    public double activate(double d) {
        return (1 / (1 + Math.exp(-d)));
    }

    @Override
    public double gradient(double d) {
        double result = activate(d);
        return (result * (1 - result));
    }

    public String toString() {
        return ("Sigmoid");
    }

}
