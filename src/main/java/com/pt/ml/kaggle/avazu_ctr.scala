package com.pt.ml.kaggle

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions.{mean, udf}
import org.apache.spark.sql.types.IntegerType

/**
 * xgb 在windows系统上没有调通
 */
object avazu_ctr {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\Program Files\\winutils\\")
    val spark = SparkSession
      .builder()
      .master("local[4]")
      .appName("kaggle_avazu_ctr")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    val dirPath = "D:\\ml-data\\avazu-ctr-prediction\\";
    import spark.implicits._
    val hour2Week = udf {
      (dateStr: String) => {
        import java.text.SimpleDateFormat
        import java.util.Calendar
        val sdf = new SimpleDateFormat("yyyyMMddHH")
        val c = Calendar.getInstance()
        c.setTime(sdf.parse("20" + dateStr))
        c.get(Calendar.DAY_OF_WEEK)
      }
    }

    val hour2hour = udf {
      (dateStr: String) => {
        dateStr.substring(6).toInt
      }
    }
    val df = spark.read.option("header", true)
      .parquet("dataset/avazu_ctr/train_data.snappy.parquet") //.csv(dirPath + "train.gz")
      .withColumn("day_of_week", hour2Week($"hour"))
      .withColumn("hourIndex", hour2hour($"hour"))
      .withColumn("label", $"click".cast(IntegerType))
      /*.filter {
        r =>
          r.getAs[Int]("label") == 1 || new Random().nextInt(1000) < 200
      }*/
      .repartition(100)
      .persist()
    df.printSchema()
    df.show(100, false)
    val trainTest = df.randomSplit(Array(0.7, 0.3))
    val trainData = trainTest(0)
    val testData = trainTest(1)

    val categoryCols = Array("C1", "banner_pos", "site_id", "site_domain", "site_category", "app_id", "app_domain", "app_category",
      "device_model", "device_type", "device_conn_type", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21")
    val categoryColsIndexer = categoryCols.map(indexer)
    val categoryColsIndex = categoryCols.map(s => s + "Index")
    val categoryColsVec = categoryCols.map(s => s + "Vec")
    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(categoryColsIndex)
      .setOutputCols(categoryColsVec)
    val assembler = new VectorAssembler()
      .setInputCols(categoryColsVec)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(100)
      .setRegParam(0.005)
      .setElasticNetParam(0.8)
    /*val xgbParam = Map(
      "num_round" -> 10
    )
    val xgbClassifier = new XGBoostClassifier(xgbParam).
      setFeaturesCol("features").
      setLabelCol("label")*/

    val dataProcessPipeline = new Pipeline()
      .setStages(categoryColsIndexer ++ Array(oneHotEncoder, assembler))

    val processAndModel = new Pipeline()
      .setStages(Array(dataProcessPipeline, lr))

    val model = processAndModel.fit(trainData)
    model.write.overwrite().save(dirPath + "lr_model")
    val probVec2Prob = udf {
      prob: Vector =>
        prob(1)
    }
    val crossEntropyLoss = udf {
      (lable: Int, pred: Double) =>
        lable * Math.log(pred) + (1 - lable) * Math.log(1 - pred)
    }
    //损失
    val loss = model.transform(testData).withColumn("probability1", probVec2Prob($"probability"))
      .select($"label", $"probability1", crossEntropyLoss($"label", $"probability1").as("loss"))
      .agg(mean("loss"))
    println("Loss损失：")
    loss.show(false)
    //提交数据
    val submissionData = spark.read.option("header", true).csv("D:\\ml-data\\avazu-ctr-prediction\\test.csv")
      .withColumn("day_of_week", hour2Week($"hour"))
      .withColumn("hourIndex", hour2hour($"hour"))

    val out = model.transform(submissionData).withColumn("probability1", probVec2Prob($"probability"))
    out.select("id", "probability1").repartition(1)
      .write.mode(SaveMode.Overwrite)
      .csv(dirPath + "/submission_result")
  }

  def indexer(str: String): StringIndexer = {
    new StringIndexer()
      .setInputCol(str)
      .setOutputCol(str + "Index")
      .setHandleInvalid("keep")
  }
}
