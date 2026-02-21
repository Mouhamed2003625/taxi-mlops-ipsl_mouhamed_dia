from pyspark import pipelines as dp

@dp.table(
    comment="Raw taxi trip data ingested from cloud storage using Auto Loader"
)
def bronze_taxi_trips():
    """
    Bronze layer: Raw taxi trip data ingestion
    Reads yellow taxi trip data from volume using Auto Loader with automatic schema inference
    """
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.inferColumnTypes", "true")
        .load("/Volumes/taxi_mlops_prod/taxi_analytics/yellowdata")
    )
