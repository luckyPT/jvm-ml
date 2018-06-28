package com.pt.ml.deeplearning;

import com.pt.ml.deeplearning.nlp.Seq2TokensByDelimiter;
import com.pt.ml.deeplearning.nlp.Word2VecDeeplearning4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * LSTM是RNN的一种，主要输入是序列化数据的场景，如：文本、语音、股票、天气等！
 * 特点是每一次的输入是本个时间步的输入和上一个时间步的输出；
 * 参数：对于LSTM层，输入是X维，输出是Y维则会有 （（X+Y）×Y+Y）×4个参数分别对应
 * -    输入，遗忘门、输入门、输出门使用。
 * 具体计算方式参考：http://colah.github.io/posts/2015-08-Understanding-LSTMs/
 * LSTM做文本二分类的时候，输出的label是2行，多列的INDArray，需要根据label的maskArray确定选择哪一列
 */
public class LstmClassification {
    public static void main(String[] args) throws Exception {
        int featureNum = 100;
        int labelNum = 2;
        List<String> trainData = FileUtils.readLines(new File("/home/panteng/IdeaProjects/jvm-ml/" +
                "dataset/word2vec-nlp-tutorial/fastTextTrain.txt"));

        DocsIterator train = new DocsIterator.Builder()
                .dataIter(trainData)
                .batchSize(200)
                .labels(Arrays.asList("0", "1"))//设置了labels
                .tokenizer(new Seq2TokensByDelimiter(" "))
                .totalCount(200)
                .truncateLength(100)
                .vectorSize(featureNum)
                .wordVectord(new Word2VecDeeplearning4j())
                .build();

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .updater(new RmsProp(0.001))
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new LSTM.Builder()
                        .nIn(featureNum)
                        .nOut(100)
                        .activation(Activation.SWISH)
                        .build()
                )
                /*.layer(1, new DenseLayer.Builder()
                        .nIn(200)
                        .nOut(200)
                        .activation(Activation.SWISH)
                        .build()
                )*/
                .layer(1, new RnnOutputLayer.Builder()
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .nIn(100)
                        .nOut(labelNum)
                        .build()
                )
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();
        net.setListeners(DeeplearningUI.startUI());
        System.out.println("--start train--");
        for (int i = 0; i < 5000; i++) {
            net.fit(train);
            System.out.println("epoch " + i + " complete score:" + net.score());
            if (i % 10 == 0) {
                /*Evaluation evaluation = net.evaluate(train);
                System.out.println(evaluation.stats());*/
                DataSet dataSet = train.next();

                int sampleCount = 100;
                INDArray labels = Nd4j.create(sampleCount);
                INDArray pre = Nd4j.create(sampleCount);
                for (int j = 0; j < sampleCount; j++) {
                    INDArray feature = dataSet.getFeatures().getRow(j);
                    INDArray label = dataSet.getLabels().getRow(j);
                    INDArray fmask = dataSet.getFeaturesMaskArray().getRow(j);
                    INDArray lmask = dataSet.getLabelsMaskArray().getRow(j);
                    int position = 0;
                    for (; position < lmask.shape()[1]; position++) {
                        if (lmask.getInt(0, position) == 1) {
                            break;
                        }
                    }
                    int[] shape = feature.shape();
                    //INDArray rawOut = net.output(feature.reshape(1, shape[0], shape[1]));
                    INDArray out = net.output(feature.reshape(1, shape[0], shape[1]), false, fmask, lmask);
                    labels.putScalar(j, label.getDouble(0, position));
                    pre.putScalar(j, out.getDouble(0, 0, position));
                    //System.out.println(label.getDouble(0, position) + "\t" + out.getDouble(0, 0, position));
                }
                INDArray off = labels.sub(pre);
                System.out.println(off.mul(off).sum(0, 1).mul(1.0 / sampleCount));
            }
        }
    }
}
