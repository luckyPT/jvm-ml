package com.pt.ml.deeplearning

import org.apache.spark.sql.SparkSession
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.{DataSet, DataSetPreProcessor}
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.JavaConversions._

/**
  * 本机简单测试，使用spark比不用spark速度快出1倍
  */
object CnnNeuralNetworkSpark {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
                .master("local[*]")
                .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")
        val featureNum = 784
        val labelNum = 10
        val batchSize = 10
        val learningRate = 0.001
        val data = File2DataSetIterator.csv("/home/panteng/IdeaProjects/jvm-ml/dataset/MINIST/train.csv",
            featureNum, labelNum, batchSize, 30000)
        data.setPreProcessor(new MyDataSetPreProcessor)

        val trainData = spark.sparkContext.parallelize(data.toList)

        val trainingMaster = new ParameterAveragingTrainingMaster.Builder(50)
                .build()
        val config = new NeuralNetConfiguration.Builder()
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .l2(0.0001)
                .list
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(1) //默认输入4维 batchSize depth width height
                        .nOut(20)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build)
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build)
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build)
                .layer(3, new DenseLayer.Builder()
                        .nOut(50)
                        .hasBias(true)
                        .activation(Activation.RELU)
                        .build)
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(labelNum)
                        .hasBias(true)
                        .activation(Activation.SOFTMAX)
                        .build)
                .setInputType(InputType.convolutional(28, 28, 1))
                .backprop(true).pretrain(false).build

        val sparkNet = new SparkDl4jMultiLayer(spark.sparkContext, config, trainingMaster)

        for (i <- Range(0, 100)) {
            println(System.currentTimeMillis + " Epoch " + i)
            sparkNet.fit(trainData)
        }
    }

    case class MyDataSetPreProcessor() extends DataSetPreProcessor {
        override def preProcess(toPreProcess: DataSet): Unit = {
            val batchSize = toPreProcess.getFeatures.size(0)
            val tmp = toPreProcess.getFeatures.reshape(batchSize, 1, 28, 28)
            toPreProcess.setFeatures(tmp)
        }
    }

}
