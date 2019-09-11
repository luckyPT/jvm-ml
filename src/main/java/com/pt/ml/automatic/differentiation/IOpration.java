package com.pt.ml.automatic.differentiation;

public interface IOpration {
    double forward(double x);

    double backward(double x);

    IOpration fromOp(IOpration op);
}
