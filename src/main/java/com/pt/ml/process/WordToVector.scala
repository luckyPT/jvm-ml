package com.pt.ml.process

import com.pt.ml.util.BinaryClassEvaluation
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.{StopWordsRemover, Word2Vec}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession

/**
  * https://blog.csdn.net/a819825294/article/details/52438625?ticket=ST-27206-CNiCr4GXfbQMifpwVGZr-passport.csdn.net
  * 使用一个向量表示一个单词,更进一步 可以用所有单词向量的平均值表示整个句子;
  * word2vec有两种类型：skip-gram和cbow；前者是根据某个词 预测 其周围的词，后者是根据周围词，预测某个词；
  * 两种建模方式：hierarchical softmax和Negative Sample的求解方式（一共四种组合）;前者计算时输出层采用的是哈夫曼树，
  * 后者计算时输出层采用的是
  * 无论是哪种模型，目标都不是求解词向量，词向量只是模型的一个副产物;模型本身是为了预测出现的词;
  *
  * spark基于softmax实现了skim model;并且在spark中每个词对应两个向量，一个做为上下文时使用，一个作为预测词时使用
  * spark基于训练的模型，可以在字典集合内查找距离某个词最近的N个词;可以利用这个初步判断词向量训练是否适合
  * 调优：窗口大小、向量长度、迭代次数、学习率
  * 效果评价：1-降维可视化  2-看最终损失   3-看后面的使用效果（如特征加入到监督学习中的效果） 4-看看近似词
  * 目前经过粗略实验，没有使用TF-IDF效果好（也许可以尝试将TF-IDF和WordToVector结合会有更好的效果）
  *
  */
object WordToVector {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession
                .builder()
                .master("local[*]")
                .getOrCreate()

        val sc = spark.sparkContext
        import spark.implicits._
        val labelReview = sc.textFile("/home/panteng/IdeaProjects/jvm-ml/dataset/word2vec-nlp-tutorial/labeledTrainData.tsv").collect {
            case str if str.split("\t").length == 3 && !str.contains("id\tsentiment\treview") =>
                val Array(_, label, review) = str.split("\t")
                val sentences = review.replaceAll("\"", "")
                        .replaceAll("\\\\", "")
                        .replaceAll("<br />", "")
                (label.toDouble, sentences, sentences.split(" ").toSeq)
        }.toDF("label", "sentence", "rawCorpus")
        val Array(trainData, testData) = labelReview.randomSplit(Array(0.8, 0.2))

        //去除停用词 可以通过.setStopWords()设置停用词
        val remover = new StopWordsRemover()
                .setInputCol("rawCorpus")
                .setOutputCol("corpus")

        val word2vec = new Word2Vec()
                .setInputCol("corpus")
                .setMaxIter(5)
                .setVectorSize(10)
                .setWindowSize(12) //这个适当大点好
                //                .setStepSize(0.01) //学习率
                .setMinCount(10) //太小 太大都不合适
                .setOutputCol("features")

        val lsvc = new LinearSVC()
                .setMaxIter(50)
                .setRegParam(0.5)
        val pipeline = new Pipeline()
                .setStages(Array(remover, word2vec, lsvc))

        val model = pipeline.fit(trainData)
        val pre = model.transform(testData)
        val preRdd = pre.select("label", "rawPrediction")
                .rdd
                .map(row => (1.0 / (1 + math.pow(Math.E, row.getAs[DenseVector](1).apply(0))), row.getDouble(0)))
        BinaryClassEvaluation.showPrecisionRecallCurve(preRdd)

    }
}
