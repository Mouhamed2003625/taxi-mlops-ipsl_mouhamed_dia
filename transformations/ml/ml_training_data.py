from pyspark import pipelines as dp
from pyspark.sql import functions as F

@dp.materialized_view(
    comment="ML training dataset with features for fare amount prediction model"
)
def ml_training_data():
    """
    ML Training Data Preparation
    Prepares features from silver layer for fare prediction model:
    - Target variable: total_amount
    - Features: trip characteristics, time features, location features
    - Filters out invalid records
    - Adds train/test split indicator
    """
    return (
        spark.read.table("silver_taxi_features")
        .filter("""
            trip_distance > 0 AND trip_distance < 100 AND
            trip_duration_minutes > 0 AND trip_duration_minutes < 180 AND
            total_amount > 0 AND total_amount < 500 AND
            fare_amount > 0 AND
            passenger_count > 0 AND passenger_count <= 6
        """)
        .select(
            # Target variable
            F.col("total_amount").alias("target_total_amount"),
            
            # Trip features
            "trip_distance",
            "trip_duration_minutes",
            "speed_mph",
            "passenger_count",
            
            # Time features
            "pickup_hour",
            "pickup_day_of_week",
            "time_of_day",
            
            # Location features
            "PULocationID",
            "DOLocationID",
            "is_airport_pickup",
            "is_airport_dropoff",
            
            # Payment and rate features
            "payment_type",
            "RatecodeID",
            
            # Date for partitioning
            "pickup_date"
        )
        # Add train/test split (80/20 split based on hash)
        .withColumn("is_training", (F.hash("pickup_date", "PULocationID") % 100) < 80)
    )
