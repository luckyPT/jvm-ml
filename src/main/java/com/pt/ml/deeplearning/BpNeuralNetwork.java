package com.pt.ml.deeplearning;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;

public class BpNeuralNetwork {
    public static void main(String[] args) throws Exception {
        double learningRate = 0.01;
        int featureNum = 4;
        int labelNum = 3;
        int layer1CellCount = 10;
        DataSetIterator data = File2DataSetIterator.svm("/home/panteng/IdeaProjects/jvm-ml/dataset/iris.libsvm", featureNum, labelNum, 20);
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .l2(0.00001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(featureNum)
                        .nOut(layer1CellCount)
                        .hasBias(true)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(layer1CellCount)
                        .nOut(labelNum)
                        .activation(Activation.SOFTMAX)
                        .hasBias(true)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(1));
        System.out.println("-----------start train-----------");
        for (int i = 0; i < 1000; i++) {
            if (i % 50 == 0) {
                System.out.println("Epoch " + i + " score=" + model.score());
            }
            model.fit(data);
        }

        data = File2DataSetIterator.svm("/home/panteng/IdeaProjects/jvm-ml/dataset/iris.libsvm", featureNum, labelNum, 20);
        Evaluation eval = new Evaluation(labelNum); //create an evaluation object with 10 possible classes
        for (int i = 0; i < 100 && data.hasNext(); i++) {
            DataSet next = data.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }
        System.out.println(eval.stats());
    }
}
