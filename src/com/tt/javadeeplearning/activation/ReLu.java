package com.tt.javadeeplearning.activation;

import java.io.Serializable;

public class ReLu implements Activation, Serializable {

    @Override
    public double activate(double d) {
        return (Math.max(0, d));
    }

    @Override
    public double gradient(double d) {
        return (d > 0 ? 1 : 0);
    }

    public String toString() {
        return ("ReLu");
    }

}
