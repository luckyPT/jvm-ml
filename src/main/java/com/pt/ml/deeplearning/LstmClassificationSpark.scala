package com.pt.ml.deeplearning

import java.io.File
import java.util
import java.util.Map

import com.pt.ml.deeplearning.nlp.{Seq2TokensByDelimiter, Word2VecDeeplearning4j, Word2VecFastText}
import org.apache.commons.io.FileUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.deeplearning4j.nn.conf.{GradientNormalization, MultiLayerConfiguration, NeuralNetConfiguration, WorkspaceMode}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.{AdaDelta, Adam, RmsProp, Sgd}
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * 使用spark时，不能实时可视化，只能将可视化所需要的文件暂时存储起来 然后再可视化
  * https://deeplearning4j.org/visualization
  *
  * 相同的网络模型，但是好像没有不用spark训练的好
  */
object LstmClassificationSpark {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
                .master("local[*]")
                .getOrCreate()

        spark.sparkContext.setLogLevel("ERROR")
        val featureNum = 100
        val labelNum = 2
        val trainingMaster = new ParameterAveragingTrainingMaster.Builder(50).build()

        val config = new NeuralNetConfiguration.Builder()
                .seed(0)
                .updater(new Adam(2e-2))
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .trainingWorkspaceMode(WorkspaceMode.SEPARATE).inferenceWorkspaceMode(WorkspaceMode.SEPARATE)   //https://deeplearning4j.org/workspaces
                .list()
                .layer(0, new LSTM.Builder().nIn(100).nOut(256)
                        .activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
                .pretrain(false).backprop(true).build()

        val sparkNet = new SparkDl4jMultiLayer(spark.sparkContext, config, trainingMaster)
        val trainData = FileUtils.readLines(new File("/home/panteng/IdeaProjects/jvm-ml/dataset/word2vec-nlp-tutorial/fastTextTrain.txt"))
        val train = new DocsIterator.Builder()
                .dataIter(trainData)
                .batchSize(500)
                .labels(util.Arrays.asList("0", "1"))
                .tokenizer(new Seq2TokensByDelimiter(" "))
                .totalCount(500)
                .truncateLength(500)
                .vectorSize(featureNum)
                .wordVectord(new Word2VecDeeplearning4j())
                .build()

        val batchData = if (train.cursor() < train.totalExamples()) {
            train.next()
        } else {
            train.reset()
            train.next()
        }
        val batchDataRdd = spark.sparkContext.parallelize(Seq(batchData)).persist(StorageLevel.MEMORY_AND_DISK_2)
        for (i <- Range(0, 50000)) {
            sparkNet.fit(batchDataRdd)
            if (i % 10 == 0) {
                val evaluation = sparkNet.evaluate(batchDataRdd)
                System.out.println(evaluation.stats)
            }
            println(System.currentTimeMillis + " Epoch " + i + " score=" + sparkNet.getScore)
        }
    }

}
