from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TesteWSL").getOrCreate()

df = spark.createDataFrame([("Alice", 30), ("Bob", 25)], ["nome", "idade"])
df.show()

spark.stop()
