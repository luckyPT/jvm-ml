package com.pt.ml.deeplearning.nlp;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface IWord2Vec {
    public int getVecSize();

    public INDArray getVec(String word);

    public boolean hasWords(String word);
}
