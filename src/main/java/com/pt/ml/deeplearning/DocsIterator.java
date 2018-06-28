package com.pt.ml.deeplearning;

import com.pt.ml.deeplearning.nlp.ISeq2Tokens;
import com.pt.ml.deeplearning.nlp.IWord2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

public class DocsIterator implements DataSetIterator {
    private final List<String> data;
    private final IWord2Vec wordVectors;
    private final ISeq2Tokens tokenizer;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;
    private final int totalCount;
    private final List<String> labels;

    private int cursor = 0;

    private DocsIterator(IWord2Vec wordVectors,
                         int batchSize,
                         int truncateLength,
                         ISeq2Tokens tokenizer,
                         List<String> data,
                         int totalCount,
                         List<String> labels) {
        this.data = data;
        this.labels = labels;
        this.totalCount = totalCount;
        this.batchSize = batchSize;
        this.vectorSize = wordVectors.getVecSize();
        this.wordVectors = wordVectors;
        this.truncateLength = truncateLength;
        this.tokenizer = tokenizer;
    }

    @Override
    public DataSet next(int num) {
        if (cursor > this.totalCount) {
            throw new NoSuchElementException();
        }
        try {
            return nextDataSet(num);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private DataSet nextDataSet(int num) throws Exception {
        long startTime = System.currentTimeMillis();
        List<String> news = new ArrayList<>(num);
        int[] category = new int[num];

        for (int i = 0; i < num && cursor < data.size(); i++) {
            String[] sampleData = data.get(cursor).split("\t");
            news.add(sampleData[1]);
            category[i] = Integer.parseInt(sampleData[0].replaceAll("__label__", ""));
            cursor++;
        }

        List<List<String>> allTokens = new ArrayList<>(news.size());
        int maxLength = 0;
        for (String s : news) {
            List<String> tokens = tokenizer.getTokens(s);
            List<String> tokensFiltered = new ArrayList<>();
            for (String t : tokens) {
                if (wordVectors.hasWords(t)) {
                    tokensFiltered.add(t);
                }
            }
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength, tokensFiltered.size());
        }
        if (maxLength > truncateLength) {
            maxLength = truncateLength;
        }
        INDArray features = Nd4j.create(news.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(news.size(), this.labels.size(), maxLength);
        INDArray featuresMask = Nd4j.zeros(news.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(news.size(), maxLength);
        int[] temp = new int[2];
        for (int i = 0; i < news.size(); i++) {
            List<String> tokens = allTokens.get(i);
            temp[0] = i;
            //Get word vectors for each word in news, and put them in the training data
            for (int j = 0; j < tokens.size() && j < maxLength; j++) {
                String token = tokens.get(j);
                INDArray vector = wordVectors.getVec(token);
                features.put(new INDArrayIndex[] {point(i),
                        all(),
                        point(j)}, vector);

                temp[1] = j;
                featuresMask.putScalar(temp, 1.0);
            }
            int idx = category[i];
            int lastIdx = Math.min(tokens.size(), maxLength);
            labels.putScalar(new int[] {i, idx, lastIdx - 1}, 1.0);
            labelsMask.putScalar(new int[] {i, lastIdx - 1}, 1.0);
            //System.out.println("------准备数据耗时：" + (System.currentTimeMillis() - startTime));
        }
        DataSet ds = new DataSet(features, labels, featuresMask, labelsMask);

        return ds;
    }

    @Override
    public int totalExamples() {
        return this.totalCount;
    }

    @Override
    public int inputColumns() {
        return this.vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return this.labels.size();
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return this.batchSize;
    }

    @Override
    public int cursor() {
        return this.cursor;
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public List<String> getLabels() {
        return this.labels;
    }

    @Override
    public boolean hasNext() {
        return cursor < numExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    public static class Builder {
        private List<String> dataIter;
        private IWord2Vec wordVectors;
        private ISeq2Tokens tokenizer;
        private int batchSize;
        private int vectorSize;
        private int truncateLength;
        private int totalCount;
        private List<String> labels;

        public Builder() {
        }

        public DocsIterator.Builder dataIter(List<String> iterator) {
            this.dataIter = iterator;
            return this;
        }

        public DocsIterator.Builder wordVectord(IWord2Vec word2Vec) {
            this.wordVectors = word2Vec;
            return this;
        }

        public DocsIterator.Builder tokenizer(ISeq2Tokens seq2Tokens) {
            this.tokenizer = seq2Tokens;
            return this;
        }

        public DocsIterator.Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public DocsIterator.Builder vectorSize(int vectorSize) {
            this.vectorSize = vectorSize;
            return this;
        }

        public DocsIterator.Builder truncateLength(int truncateLength) {
            this.truncateLength = truncateLength;
            return this;
        }

        public DocsIterator.Builder totalCount(int totalCount) {
            this.totalCount = totalCount;
            return this;
        }

        public DocsIterator.Builder labels(List<String> labels) {
            this.labels = labels;
            return this;
        }

        public DocsIterator build() {
            return new DocsIterator(wordVectors, batchSize, truncateLength, tokenizer, dataIter, totalCount, labels);
        }
    }
}
