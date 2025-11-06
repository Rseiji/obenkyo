from pyspark.sql import SparkSession


# To run: spark-submit <filename> --jars <jdbc driver path>
if __name__ == "__main__":
    spark = SparkSession.builder.appName("StreamingTest").getOrCreate()
    jsonschema = "nome STRING, postagem STRING, data INT"

    # Declaration of a dataframe got from streamed data
    # schema must be well defined, no inference possible
    df = spark.readStream.json(
        "/home/ubuntu/obenkyo/raw_data/spark_course_udemy/streaming_test/", schema=jsonschema
    )

    # Store streaming session data. "Memory"
    directory = "/home/ubuntu/tmp_streaming/"

    def update_postgres(df, batch_id):
        (
            df
            .write
            .format("jdbc")
            .option("url", "jdbc:postgresql://localhost:5432/posts")
            .option("dbtable", "posts")
            .option("user", "postgres")
            .option("password", "123")
            .option("driver", "org.postgresql.Driver")
            .mode("append")
            .save()
        )

    stcal = (
        df
        .writeStream
        .foreachBatch(update_postgres)  # What to do for each batch
        .outputMode("append")
        .trigger(processingTime="5 seconds")
        .option("checkpointLocation", directory)
        .start()
    )

    stcal.awaitTermination()
