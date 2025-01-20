package com.tt.javadeeplearning.activation;

import java.io.Serializable;

public interface Activation {

    double activate(double d);

    double gradient(double d);

}
