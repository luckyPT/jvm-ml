package com.pt.ml.automatic.differentiation;

public class SigmoidOp extends AbstarctOp {
    @Override
    public double forward(double x) {
        return 1 / (1 + Math.pow(Math.E, 0 - x));
    }

    @Override
    public double backward(double x) {
        double fv = forward(x);
        return fv * (1 - fv);
    }
}
