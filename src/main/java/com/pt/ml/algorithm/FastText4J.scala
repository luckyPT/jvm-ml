package com.pt.ml.algorithm

import java.io.{File, FileWriter, PrintWriter}

import com.github.jfasttext.JFastText
import org.apache.spark.sql.SparkSession

import scala.io.Source
import scala.util.Try

/**
  * https://github.com/facebookresearch/fastText
  * https://fasttext.cc
  * 速度确实很快
  * 并且在精确度上也比TF-IDF 高出一些（>0.88 测试集上达到0.87）
  *
  * fastText包含两个功能：
  * - 单词的向量表示(skipgram或cbow)
  *   - -dim词向量维度（一般词越多，这个值应该越大;通常100～300之间）
  *   - -minn/-maxn subString的最小字符数和最大字符数，当一个单词没有出现在语料中时，将这个词截断成subString然后求和
  *   - -epoch 循环迭代的次数
  *   - -lr 学习率 learn rate
  *   - 可以通过计算向量夹角的cons值计算相似性，有交互式的C语言的API，没有实现java对应的API;
  *   - 还可以计算相对值，（中国 - 北京 + 巴黎）应该等于法国
  * - 句子的分类
  *   - 增加迭代次数
  *   - -lr 可以适当调大或者调低
  *   - -wordNgrams 对于特别关注语序的，应当设置大一些
  *   - -loss hs加速训练
  *
  *
  */
object FastText4J {
    def main(args: Array[String]): Unit = {
        fastTextTrain()
        test()
        //submission()
    }

    def trainWord2Vec(): Unit = {
        val jft = new JFastText
        jft.runCmd(Array[String]("skipgram",
            "-input", "dataset/word2vec-nlp-tutorial/unlabelTrainData",
            "-output", "dataset/word2vec-nlp-tutorial/fastText/skip-gram.model",
            "-bucket", "20",
            "-dim", "10",
            "-minn", "3",
            "-maxn", "6",
            "-wordNgrams", "5",
            "-minCount", "5"))
    }

    def fastTextTrain(): Unit = {
        val jft = new JFastText
        jft.runCmd(Array[String]("supervised",
            "-input", "dataset/word2vec-nlp-tutorial/fastTextTrain.txt",
            "-output", "dataset/word2vec-nlp-tutorial/fastText/fastText.model",
            "-lr", "0.5",
            "-lrUpdateRate", "100",
            "-dim", "300",
            "-ws", "10",
            "-wordNgrams", "1",//会降低训练速度，并且没发现什么效果
            "-neg", "10",
            "-loss", "softmax",
            "-epoch", "30"
        ))
        println("train complete")
    }

    def test(): Unit = {
        val jft = new JFastText
        jft.loadModel("dataset/word2vec-nlp-tutorial/fastText/fastText.model.bin")
        val preAndLabel = Source.fromFile("dataset/word2vec-nlp-tutorial/fastTextTest.txt")
                .getLines()
                .map {
                    str =>
                        val labelAndText = str.split("\t")
                        val probLabel = jft.predictProba(labelAndText(1))
                        val pre = probLabel.label.replaceAll("__label__", "").toInt
                        val label = labelAndText(0).replaceAll("__label__", "").toInt
                        (pre, label)
                }.toSeq

        val spark = SparkSession
                .builder()
                .master("local[*]")
                .getOrCreate()
        val sc = spark.sparkContext
        val preAndLabelRdd = sc.parallelize(preAndLabel)
        val count = preAndLabel.size
        val accuracy = preAndLabelRdd.map {
            case (pre, label) =>
                if (pre > 0.5 && label > 0.5) {
                    1
                } else if (pre <= 0.5 && label <= 0.5) {
                    1
                } else {
                    0
                }
        }.collect().sum
        println("accuracy:" + accuracy * 1.0 / count)
    }

    def submission(): Unit = {
        val jft = new JFastText
        jft.loadModel("dataset/word2vec-nlp-tutorial/fastText/fastText.model.bin")
        val subFileWriter =
            new FileWriter(new File("dataset/word2vec-nlp-tutorial/fastText/submision2.txt"))

        Source.fromFile("dataset/word2vec-nlp-tutorial/testData.tsv")
                .getLines()
                .foreach {
                    str =>
                        val idAndText = str.split("\t")
                        if (idAndText.size == 2 && !"\"5147_1\"".equals(idAndText(0))) {
                            val probLabel = jft.predictProba(idAndText(1).replaceAll("\"", "")
                                    .replaceAll("\\\\", "")
                                    .replaceAll("<br />", ""))

                            val pre = probLabel.label.replaceAll("__label__", "").toInt
                            subFileWriter.write(Seq(idAndText(0), pre).mkString(",") + "\n")
                        } else {
                            println(idAndText(0) + "\t" + idAndText(1))
                        }
                }
        subFileWriter.flush()
        subFileWriter.close()
    }

    def fastTextFormat(): Unit = {
        val trainWriter = new PrintWriter(new FileWriter(new File("dataset/word2vec-nlp-tutorial/fastTextTrain.txt")))
        val testWriter = new PrintWriter(new FileWriter(new File("dataset/word2vec-nlp-tutorial/fastTextTest.txt")))
        Source.fromFile("dataset/word2vec-nlp-tutorial/labeledTrainData.tsv")
                .getLines()
                .map {
                    str =>
                        val array = str.split("\t")
                        Seq("__label__" + array(1), array(2).replaceAll("\"", "")
                                .replaceAll("\\\\", "")
                                .replaceAll("<br />", "")
                        ).mkString("\t")
                }
                .foreach(str => {
                    if (!str.contains("sentiment\treview"))
                        if (str.hashCode % 100 < 80)
                            trainWriter.println(str)
                        else
                            testWriter.println(str)

                })
        trainWriter.flush()
        testWriter.flush()
        Try(trainWriter.close())
        Try(testWriter.close())
        println("----success----")
    }

    def getUnlabeledTrainData(): Unit = {
        val fileWriter = new FileWriter(new File("dataset/word2vec-nlp-tutorial/unlabelTrainData"))
        Source.fromFile("dataset/word2vec-nlp-tutorial/fastTextTrain.txt")
                .getLines()
                .map {
                    str =>
                        str.split("\t")(1)
                }
                .foreach {
                    str =>
                        fileWriter.write(str + "\n")
                }
        fileWriter.flush()
        fileWriter.close()
    }
}
