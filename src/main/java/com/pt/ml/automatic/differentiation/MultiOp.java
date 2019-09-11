package com.pt.ml.automatic.differentiation;

public class MultiOp extends AbstarctOp {
    double w;

    public MultiOp(double w) {
        this.w = w;
    }

    @Override
    public double forward(double x) {
        return w * x;
    }

    @Override
    public double backward(double x) {
        return w;
    }

}
