package com.pt.ml.deeplearning;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * 可以访问localhost:9000查看训练过程
 *
 * @see DeeplearningUI
 */
public class CnnNeuralNetwork {
    public static void main(String[] args) throws Exception {
        int featureNum = 784;
        int labelNum = 10;
        int batchSize = 10;
        double learningRate = 0.001;
        DataSetPreProcessor preProcessor = new MyDataSetPreProcessor();
        DataSetIterator trainData = File2DataSetIterator.csv("/home/panteng/IdeaProjects/jvm-ml/dataset/MINIST/train.csv",
                featureNum, labelNum, batchSize, 30000);
        trainData.setPreProcessor(preProcessor);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .l2(0.0001)
                .list()
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(1) //默认输入4维 batchSize depth width height
                        .nOut(20)//设置卷积层的纵深
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nOut(50)
                        .hasBias(true)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(labelNum)
                        .hasBias(true)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(28, 28, 1))
                .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();
        net.setListeners(DeeplearningUI.startUI());
        DataSetIterator testData = File2DataSetIterator.csv("/home/panteng/IdeaProjects/jvm-ml/dataset/MINIST/train.csv",
                featureNum, labelNum, batchSize, 1);
        testData.setPreProcessor(preProcessor);
        System.out.println("start train...");
        for (int i = 0; i < 100; i++) {
            net.fit(trainData);
            System.out.println(System.currentTimeMillis() + " Epoch " + i);

            //evaluation
            if (i % 10 == 0) {
                Evaluation eval = new Evaluation(labelNum);
                for (int j = 0; j < 100; j++) {
                    DataSet next = testData.next();
                    INDArray output = net.output(next.getFeatureMatrix());
                    eval.eval(next.getLabels(), output);
                }
                System.out.println(eval.stats());
            }
        }

    }

    static class MyDataSetPreProcessor implements DataSetPreProcessor {
        @Override
        public void preProcess(DataSet toPreProcess) {
            int batchSize = toPreProcess.getFeatures().size(0);
            INDArray tmp = toPreProcess.getFeatures().reshape(batchSize, 1, 28, 28);
            toPreProcess.setFeatures(tmp);
        }
    }
}
