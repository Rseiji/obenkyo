from pyspark.sql import SparkSession


# To run: spark-submit <filename>
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

    stcal = (
        df
        .writeStream
        .format("console")
        .outputMode("append")  # Append new rows. Don't try to rewrite everything from the beginning (i.e. mode "complete")
        .trigger(processingTime="5 seconds")  # Checks for each 5 seconds
        .option("checkpointLocation", directory)
        .start()
    )

    stcal.awaitTermination()
