package com.pt.ml.process

import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession

/**
  * 在文本挖掘中，用来衡量一个词对于一个文档的重要性，或者有别于其他文档的权重
  * 思考：对于有监督分类，可以将一类文档合并为一个文档，这样可以计算某个词对这一类的重要性
  * - TF：某个词在特定文档中出现的次数，一般为了解决各个文档长短不一造成的负面，常常除以文档长度（词数）；
  * -    但即时这种计算方式，仍然偏重于那些常用词；如 a、the、of等
  * - IDF：log(文档总数/包含词条的文档数)  为了防止0出现，分子分母都加1
  * - 输入：矩阵（每个词代表一列）
  * - 输出：向量（代表每个词的IDF值）
  *
  * 生成词频：HashingTF 和 CountVectorizer两种方式
  * HashingTF
  * - 实际实现并不十分科学，首先是没有考虑句子长度
  * - 其次计算过程先计算词的hash值，然后取模映射到列。在setNumFeatures很小的时候，误差很大
  * - 降低这种误差，只能是将setNumFeatures增大
  * CountVectorizer
  * - 一种标准的计算方式计算TF，仍然没考虑句子长度的影响;
  * 扩展方案：
  * - 为了按照标准的TF-IDF进行计算，需要修正CountVectorizer的转换结果，
  * - 修正之后的结果继续使用IDF 不受影响,IDF源码如下；
  *         doc match {
  *              case SparseVector(size, indices, values) =>
  *               val nnz = indices.length
  *                var k = 0
  *                while (k < nnz) {
  *                  if (values(k) > 0) {
  *                    df(indices(k)) += 1L
  *                  }
  *                  k += 1
  *                }
  *              case DenseVector(values) =>
  *                val n = values.length
  *                var j = 0
  *                while (j < n) {
  *                  if (values(j) > 0.0) {
  *                    df(j) += 1L
  *                  }
  *                  j += 1
  *        }
  */
object TfIdf {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder().master("local[*]").getOrCreate()
        val sqlContext = spark.sqlContext
        val sc = spark.sparkContext

        val sentenceData = spark.createDataFrame(Seq(
            (0.0, "京东 淘宝 小米 京东"),
            (0.0, "京东 百度 天猫 百度"),
            (1.0, "苹果 华为 亚马逊 联想")
        )).toDF("label", "sentence")
        //会自动将大写变为小写
        val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
        val wordsData = tokenizer.transform(sentenceData)

        val hashingTF = new HashingTF()
                .setInputCol("words")
                .setOutputCol("rawFeatures")
                .setNumFeatures(4096)
        val hashTf = hashingTF.transform(wordsData)
        hashTf.show(false)

        val cvModel: CountVectorizerModel = new CountVectorizer()
                .setInputCol("words")
                .setOutputCol("rawFeatures")
                .setVocabSize(4096)
                .setMinDF(1)
                .fit(wordsData)
        val countVectorizerTf = cvModel.transform(wordsData)
        countVectorizerTf.show(false)
        println(countVectorizerTf.schema)
        import sqlContext.implicits._

        //计算新输入句子的向量
        val preSentences = Seq(Array("京东", "百度", "不存在")).toDF("words")
        preSentences.show(false)
        val pre = cvModel.transform(preSentences)
        pre.show(false)

        //IDF
        val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
        val idfModel = idf.fit(hashTf)
        val rescaledData = idfModel.transform(hashTf)
        rescaledData.select("label", "features").show(false)
    }
}
