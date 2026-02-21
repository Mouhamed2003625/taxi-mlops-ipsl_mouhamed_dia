from pyspark import pipelines as dp
from pyspark.sql import functions as F

@dp.table(
    comment="Cleaned taxi trip data with engineered features for ML model training"
)
# Expectations sur les données brutes (suppression des enregistrements invalides)
@dp.expect_all_or_drop({
    "valid_trip_distance": "trip_distance > 0 AND trip_distance < 100",
    "valid_fare": "fare_amount > 0 AND fare_amount < 500",
    "valid_passenger_count": "passenger_count > 0 AND passenger_count <= 6",
    "valid_timestamps": "tpep_pickup_datetime < tpep_dropoff_datetime"
})
# Expectations supplémentaires (les enregistrements qui échouent sont conservés mais signalés)
@dp.expect_all({
    "reasonable_tip": "tip_amount >= 0 AND tip_amount < 100",
    "reasonable_total": "total_amount > 0 AND total_amount < 1000",
    # --- Nouvelle feature : prix par mile ---
    "valid_price_per_mile": "price_per_mile >= 0",
    # --- Nouvelle feature : type de paiement ---
    "valid_payment_type_label": "payment_type_label IS NOT NULL",
    # --- Nouvelle feature : indicateur week-end ---
    "valid_is_weekend": "is_weekend IN (0, 1)"
})
def silver_taxi_features():
    """
    Silver layer: Feature engineering for ML

    This function reads raw taxi trips from the bronze table and applies:
    - Data cleaning & filtering (via expectations)
    - Feature engineering (time, distance, speed, airport flags, etc.)
    - Three new features added for Part 2 of the project:
        1. price_per_mile : fare amount divided by trip distance.
        2. payment_type_label : human-readable payment method.
        3. is_weekend : boolean flag for Saturday/Sunday pickups.
    """
    return (
        spark.readStream.table("bronze_taxi_trips")
        .filter("VendorID IS NOT NULL")
        
        # --- 1. Trip duration (minutes) ---
        .withColumn(
            "trip_duration_minutes",
            (F.unix_timestamp("tpep_dropoff_datetime") - 
             F.unix_timestamp("tpep_pickup_datetime")) / 60
        )
        
        # --- 2. Average speed (mph) ---
        .withColumn(
            "speed_mph",
            F.when(
                F.col("trip_duration_minutes") > 0,
                (F.col("trip_distance") / F.col("trip_duration_minutes")) * 60
            ).otherwise(0)
        )
        
        # --- 3. Time-based features ---
        .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("pickup_day_of_week", F.dayofweek("tpep_pickup_datetime"))
        .withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
        
        # --- 4. Time of day category ---
        .withColumn(
            "time_of_day",
            F.when((F.col("pickup_hour") >= 6) & (F.col("pickup_hour") < 12), "morning")
            .when((F.col("pickup_hour") >= 12) & (F.col("pickup_hour") < 18), "afternoon")
            .when((F.col("pickup_hour") >= 18) & (F.col("pickup_hour") < 22), "evening")
            .otherwise("night")
        )
        
        # --- 5. Airport indicators (JFK=132, LaGuardia=138, Newark=1) ---
        .withColumn(
            "is_airport_pickup",
            F.col("PULocationID").isin([1, 132, 138])
        )
        .withColumn(
            "is_airport_dropoff",
            F.col("DOLocationID").isin([1, 132, 138])
        )
        
        # ========== NOUVELLES FEATURES (Partie 2) ==========
        
        # --- 6. Price per mile (fare amount per unit distance) ---
        # Indicates fare efficiency; can help detect zones with different pricing.
        .withColumn(
            "price_per_mile",
            F.when(F.col("trip_distance") > 0, 
                   F.col("fare_amount") / F.col("trip_distance"))
            .otherwise(0.0)
        )
        
        # --- 7. Payment type label (human-readable) ---
        # Based on NYC TLC payment_type codes:
        # 1 = Credit card, 2 = Cash, 3 = No charge, 4 = Dispute, 5 = Unknown, 6 = Voided trip.
        .withColumn(
            "payment_type_label",
            F.when(F.col("payment_type") == 1, "credit_card")
            .when(F.col("payment_type") == 2, "cash")
            .when(F.col("payment_type") == 3, "no_charge")
            .when(F.col("payment_type") == 4, "dispute")
            .when(F.col("payment_type") == 5, "unknown")
            .when(F.col("payment_type") == 6, "voided")
            .otherwise("other")
        )
        
        # --- 8. Weekend indicator (1 = Saturday or Sunday, 0 = weekday) ---
        # Useful because travel patterns (and often pricing) differ on weekends.
        .withColumn(
            "is_weekend",
            F.when(F.col("pickup_day_of_week").isin([1, 7]), 1).otherwise(0)
        )
    )

