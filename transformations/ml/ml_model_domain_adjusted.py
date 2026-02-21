from pyspark import pipelines as dp
from pyspark.sql import functions as F

@dp.table(
    comment="ML model with 2 domain-specific adjustments"
)
def ml_model_domain_adjusted():
    """
    MODÈLE AVEC 2 RÉGLAGES SPÉCIFIQUES AU DOMAINE
    
    RÉGLAGE 1: Surcharge par aéroport
    - JFK (132): +6$
    - LaGuardia (138): +4$  
    - Newark (1): +8$
    
    RÉGLAGE 2: Tarif minimum
    - Course < 1 mile = minimum 5$
    """
    
    # Charger les données d'entraînement
    data = spark.read.table("taxi_mlops_prod.mouhamed_dia.ml_training_data")
    
    # Modèle de base (formule linéaire simple)
    base_model = data.withColumn(
        "base_prediction",
        F.lit(3.0) + 
        (F.col("trip_distance") * 2.5) + 
        (F.col("trip_duration_minutes") * 0.5) +
        F.when(F.col("time_of_day") == "evening", 2.0).otherwise(0) +
        (F.col("passenger_count") * 0.5)
    )
    
    # Appliquer les réglages domaine
    final_model = base_model.withColumn(
        "final_prediction",
        F.col("base_prediction") +
        
        # RÉGLAGE 1: Surcharge aéroport par terminal
        F.when(F.col("PULocationID") == 132, 6.0)
        .when(F.col("DOLocationID") == 132, 6.0)
        .when(F.col("PULocationID") == 138, 4.0)
        .when(F.col("DOLocationID") == 138, 4.0)
        .when(F.col("PULocationID") == 1, 8.0)
        .when(F.col("DOLocationID") == 1, 8.0)
        .otherwise(0)
    ).withColumn(
        "final_prediction",
        # RÉGLAGE 2: Minimum de course
        F.when(F.col("trip_distance") < 1, 
               F.greatest(F.col("final_prediction"), F.lit(5.0)))
        .otherwise(F.col("final_prediction"))
    )
    
    # Calculer les métriques en utilisant la colonne cible réelle
    metrics = final_model.agg(
        F.sqrt(F.avg((F.col("target_total_amount") - F.col("base_prediction")) ** 2)).alias("base_rmse"),
        F.sqrt(F.avg((F.col("target_total_amount") - F.col("final_prediction")) ** 2)).alias("final_rmse"),
        F.avg(F.abs(F.col("target_total_amount") - F.col("base_prediction"))).alias("base_mae"),
        F.avg(F.abs(F.col("target_total_amount") - F.col("final_prediction"))).alias("final_mae"),
        F.pow(F.corr("target_total_amount", "base_prediction"), 2).alias("base_r2"),
        F.pow(F.corr("target_total_amount", "final_prediction"), 2).alias("final_r2")
    )
    
    result = metrics.select(
        "base_rmse",
        "base_mae",
        "base_r2",
        "final_rmse",
        "final_mae",
        "final_r2",
        F.lit("domain_adjusted_v1").alias("model_version"),
        F.current_timestamp().alias("training_date"),
        F.lit("airport_surcharge_by_terminal, minimum_fare_5usd").alias("domain_adjustments"),
        F.round(((F.col("base_rmse") - F.col("final_rmse")) / F.col("base_rmse")) * 100, 2).alias("rmse_improvement_pct"),
        F.round(((F.col("base_mae") - F.col("final_mae")) / F.col("base_mae")) * 100, 2).alias("mae_improvement_pct"),
        F.round(F.col("final_r2") - F.col("base_r2"), 3).alias("r2_improvement")
    )
    
    return result