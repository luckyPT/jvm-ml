package com.pt.ml.deeplearning.nlp;

import com.pt.ml.deeplearning.DocsIterator;
import com.pt.ml.process.TSNEStandard;
import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.List;

/**
 * 对文档向量化之后，使用TSNE可视化 效果不太理想
 */
public class DocsClassificationTsne {
    public static void main(String[] args) throws Exception {
        /*docsToCsv("/home/panteng/IdeaProjects/jvm-ml/dataset/word2vec-nlp-tutorial/fastTextTrain.txt",
                "/home/panteng/IdeaProjects/jvm-ml/dataset/word2vec-nlp-tutorial/fastTextTrain.csv");*/
        TSNEStandard.tsneVisualization2D("/home/panteng/IdeaProjects/jvm-ml/dataset/word2vec-nlp-tutorial/fastTextTrain.csv",
                10000, true, 1000, 500, 1800);
    }

    public static void docsToCsv(String inputFile, String outputFile) throws Exception {
        int sampleCount = 2000;
        int featureNum = 100;
        int labelNum = 2;
        List<String> trainData = FileUtils.readLines(new File(inputFile));

        DocsIterator train = new DocsIterator.Builder()
                .dataIter(trainData)
                .batchSize(sampleCount)
                .labels(Arrays.asList("0", "1"))
                .tokenizer(new Seq2TokensByDelimiter(" "))
                .totalCount(sampleCount)
                .truncateLength(100)
                .vectorSize(featureNum)
                .wordVectord(new Word2VecDeeplearning4j())
                .build();

        DataSet dataSet = train.next();
        INDArray featrue = dataSet.getFeatures().reshape(sampleCount, -1);
        INDArray labels = dataSet.getLabels().sum(2).getColumn(1);

        double[][] labelFeature = Nd4j.concat(1, labels, featrue).toDoubleMatrix();
        File file = new File(outputFile);
        BufferedWriter writer = new BufferedWriter(new FileWriter(file));

        for (double[] aLabelFeature : labelFeature) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < aLabelFeature.length; j++) {
                if (j != 0) {
                    sb.append(",");
                }
                sb.append(aLabelFeature[j]);
            }
            sb.append("\n");
            writer.write(sb.toString());
        }
        writer.flush();
        writer.close();
    }
}
