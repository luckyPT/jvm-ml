package com.pt.ml.algorithm

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

object Xgboost4jSpark {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
                .master("local[8]").getOrCreate()
        val schema = new StructType(Array(
            StructField("class", StringType, true),
            StructField("sepal length", DoubleType, true),
            StructField("sepal width", DoubleType, true),
            StructField("petal length", DoubleType, true),
            StructField("petal width", DoubleType, true)))
        val rawInput = spark.read.schema(schema).csv("dataset/iris.csv")

        import org.apache.spark.ml.feature.StringIndexer
        val stringIndexer = new StringIndexer().
                setInputCol("class").
                setOutputCol("classIndex").
                fit(rawInput)
        val labelTransformed = stringIndexer.transform(rawInput).drop("class")

        import org.apache.spark.ml.feature.VectorAssembler
        val vectorAssembler = new VectorAssembler().
                setInputCols(Array("sepal length", "sepal width", "petal length", "petal width")).
                setOutputCol("features")
        val xgbInput = vectorAssembler.transform(labelTransformed).select("features", "classIndex")

        val xgbParam = Map(
            "eta" -> 0.1f,
            "missing" -> -999,
            "objective" -> "multi:softprob",
            "num_class" -> 3,
            "num_round" -> 20,
            "num_workers" -> 2
        )

        val xgbClassifier = new XGBoostClassifier(xgbParam).
                setFeaturesCol("features").
                setLabelCol("classIndex")
        val xgbClassificationModel = xgbClassifier.fit(xgbInput)
        xgbClassificationModel.save("/home/panteng/IdeaProjects/jvm-ml/xgbModel")
        spark.close()
    }
}
