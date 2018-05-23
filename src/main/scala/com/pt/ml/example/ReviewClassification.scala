package com.pt.ml.example

import com.pt.ml.util.BinaryClassEvaluation
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LinearSVC, LogisticRegression}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, Tokenizer}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession

/**
  * https://www.kaggle.com/c/word2vec-nlp-tutorial/
  * 根据评论 判断用户情绪
  * 具体流程：
  * - 分词 Tokenizer
  * - 基于词频进行向量化 CountVectorizer
  * - 加入IDF IDF
  * - LR模型
  * 以上过程统一加入到Pipeline中
  * 使用SVM 在验证集上PR交点在0.87～0.88之间；在提交测试集上，accuracy=0.859
  */
object ReviewClassification {
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
                (label.toDouble, review.replaceAll("\"", "")
                        .replaceAll("\\\\", "")
                        .replaceAll("<br />", ""))
        }.toDF("label", "sentence")
        val submissionData = sc.textFile("/home/panteng/IdeaProjects/jvm-ml/dataset/word2vec-nlp-tutorial/testData.tsv").collect {
            case str if str.split("\t").length == 2 && !str.contains("id\treview") =>
                val Array(id, review) = str.split("\t")
                (id, review.replaceAll("\"", "")
                        .replaceAll("\\\\", "")
                        .replaceAll("<br />", ""))
        }.toDF("id", "sentence")

        val Array(trainData, testData) = labelReview.randomSplit(Array(0.8, 0.2))

        val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
        val cvVectorized = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("rawFeatures")
                .setVocabSize(819200)
                .setMinDF(1)
        val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
        //构建模型，训练
        val lr = new LogisticRegression()
                .setMaxIter(20)
                .setRegParam(0.1)
                .setElasticNetParam(0)
        val lsvc = new LinearSVC()
                .setMaxIter(50)
                .setRegParam(0.5) //实际调整时，发现这个值大点好

        val pipeline = new Pipeline()
                .setStages(Array(tokenizer, cvVectorized, idf, lsvc))

        val model = pipeline.fit(trainData)
        //lr-probability  svm-rawPrediction
        val pre = model.transform(testData)
        val preRdd = pre.select("label", "rawPrediction")
                .rdd
                .map(row => (1.0 / (1 + math.pow(Math.E, row.getAs[DenseVector](1).apply(0))), row.getDouble(0)))
        BinaryClassEvaluation.showPrecisionRecallCurve(preRdd)
        //subMission
        val subPre = model.transform(submissionData)
        subPre.select("id", "prediction")
                .rdd
                .map {
                    row =>
                        Seq(row(0).asInstanceOf[String], row(1).asInstanceOf[Double].toInt).mkString(",")
                }
                .repartition(1).saveAsTextFile("/home/panteng/IdeaProjects/jvm-ml/dataset/word2vec-nlp-tutorial/svm/submission")
    }
}
