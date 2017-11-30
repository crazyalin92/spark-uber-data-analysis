/**
  * Created by ALINA on 29.11.2017.
  */

import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.TimestampType
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeans

object UberDataAnalysis {

  case class Uber(dt: String, lat: Double, lon: Double, base: String) extends Serializable

  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    //Initialize SparkSession
    val sparkSession = SparkSession
      .builder()
      .appName("spark-uber-analysis")
      .master("local[*]")
      .getOrCreate();

    import sparkSession.implicits._

    val uberDataDir = args(0)

    val schema = StructType(Array(
      StructField("dt", TimestampType, true),
      StructField("lat", DoubleType, true),
      StructField("lon", DoubleType, true),
      StructField("base", StringType, true)
    ))

    //Load Uber Data to DF
    val uberData = sparkSession.read
      .option("header", "true")
      .option("inferSchema", "false")
      .schema(schema)
      .csv(uberDataDir)
      .as[Uber]

    uberData.show()
    uberData.printSchema()

    uberData.createOrReplaceTempView("uber")

    //Get the name of company which have the maximum count of trips
    sparkSession.sql("SELECT base, COUNT(base) as cnt FROM uber GROUP BY base").show()

    //Get the dates with the maximum count of trips
    sparkSession.sql("SELECT date(dt), COUNT(base) as cnt FROM uber GROUP BY date(dt), base ORDER BY 1").show()

    // Get Feature Vectors
    val featureCols = Array("lat", "lon")
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val uberFeatures = assembler.transform(uberData)

    //Split data into training and testing data
    val Array(trainingData, testData) = uberFeatures.randomSplit(Array(0.7, 0.3), 5043)

    //Traing KMeans model
    val kmeans = new KMeans()
      .setK(20)
      .setFeaturesCol("features")
      .setMaxIter(5)

    val model = kmeans.fit(trainingData)

    println("Final Centers: ")
    model.clusterCenters.foreach(println)

    //Get Predictions
    val predictions = model.transform(testData)
    predictions.show

    predictions.createOrReplaceTempView("uber")

    predictions.select(month($"dt").alias("month"), dayofmonth($"dt").alias("day"), hour($"dt").alias("hour"), $"prediction")
      .groupBy("month", "day", "hour", "prediction")
      .agg(count("prediction")
        .alias("count"))
      .orderBy("day", "hour", "prediction").show

    //Which hours of the day and which cluster had the highest number of pickups?
    predictions.select(hour($"dt").alias("hour"), $"prediction")
      .groupBy("hour", "prediction").agg(count("prediction")
      .alias("count"))
      .orderBy(desc("count"))
      .show

    predictions.groupBy("prediction").count().show()

    sparkSession.sql("select prediction, count(prediction) as count from uber group by prediction").show

    sparkSession.sql("select hour(uber.dt) as hr,count(prediction) as ct FROM uber group By hour(uber.dt)").show

    //save uber data to json
    val res = sparkSession.sql("select dt, lat, lon, base, prediction as cid FROM uber where prediction = 1")
    res.coalesce(1).write.format("json").save("./data/uber.json")
  }
}
