package com.pt.ml.deeplearning.nlp;

import com.github.jfasttext.JFastText;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Word2VecFastText implements IWord2Vec {
    private static final JFastText jft = new JFastText();
    private static final int size;

    static {
        jft.loadModel("/home/panteng/IdeaProjects/jvm-ml/dataset/word2vec-nlp-tutorial/fastText/skip-gram.model.bin");
        size = jft.getVector("test").size();
    }

    @Override
    public int getVecSize() {
        return size;
    }

    @Override
    public INDArray getVec(String word) {
        return Nd4j.create(jft.getVector(word));
    }

    @Override
    public boolean hasWords(String word) {
        return true;
    }
}