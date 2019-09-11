package com.pt.ml.automatic.differentiation;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstarctOp implements IOpration {
    List<IOpration> inputOps = new ArrayList<>();

    @Override
    public IOpration fromOp(IOpration op) {
        this.inputOps.add(op);
        return this;
    }
}
