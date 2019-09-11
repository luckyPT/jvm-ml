package com.pt.ml.automatic.differentiation;

public class InputOp extends AbstarctOp {

    public InputOp() {
    }

    @Override
    public double forward(double x) {
        return x;
    }

    @Override
    public double backward(double x) {
        throw new RuntimeException("illegal operation for InputOp");
    }

    @Override
    public IOpration fromOp(IOpration op) {
        throw new RuntimeException("illegal operation for InputOp");
    }
}
