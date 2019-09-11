package com.pt.ml.algorithm

import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.SparkSession

object LdaCluster {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
                .appName("lda")
                .master("local[4]")
                .getOrCreate()
        // Loads data.
        val dataset = spark.read.format("libsvm").load("dataset/spark_lda/sample_lda_libsvm_data.txt")
        dataset.printSchema()

        // Trains a LDA model.
        val lda = new LDA().setK(10).setMaxIter(10)
        val model = lda.fit(dataset)

        val ll = model.logLikelihood(dataset)
        val lp = model.logPerplexity(dataset)
        println(s"The lower bound on the log likelihood of the entire corpus: $ll")
        println(s"The upper bound on perplexity: $lp")

        // Describe topics.
        val topics = model.describeTopics(3)
        println("The topics described by their top-weighted terms:")
        topics.show(false)

        // Shows the result.
        val transformed = model.transform(dataset)
        transformed.show(false)
    }
}
