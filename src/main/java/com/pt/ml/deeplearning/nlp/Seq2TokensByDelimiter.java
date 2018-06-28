package com.pt.ml.deeplearning.nlp;

import java.util.Arrays;
import java.util.List;

public class Seq2TokensByDelimiter implements ISeq2Tokens {
    private final String delimiter;

    public Seq2TokensByDelimiter(String delimiter) {
        this.delimiter = delimiter;
    }

    @Override
    public List<String> getTokens(String seq) {
        return Arrays.asList(seq.split(delimiter));
    }
}
