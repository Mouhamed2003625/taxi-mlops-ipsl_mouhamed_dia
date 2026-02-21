from pyspark import pipelines as dp
from pyspark.sql import functions as F

@dp.table(
    comment="Improved ML model with 3 enhancements using Part 2 features"
)
def ml_model_improved():
    """
    MODÈLE AMÉLIORÉ AVEC 3 NOUVELLES FONCTIONNALITÉS
    """
    
    # 1. Charger les données d'entraînement
    training_data = spark.read.table("taxi_mlops_prod.mouhamed_dia.ml_training_data")
    
    # 2. Charger les nouvelles features de silver
    silver_features = spark.read.table("taxi_mlops_prod.mouhamed_dia.silver_taxi_features")
    
    # 3. Sélectionner les colonnes pour la jointure (clés + nouvelles features)
    new_features = silver_features.select(
        "PULocationID",
        "DOLocationID",
        "pickup_date",
        "pickup_hour",
        "price_per_mile",
        "payment_type_label",
        "is_weekend"
    ).distinct()
    
    # 4. Joindre les nouvelles features (left join pour ne pas perdre d'enregistrements)
    enhanced_data = training_data.join(
        new_features,
        on=["PULocationID", "DOLocationID", "pickup_date", "pickup_hour"],
        how="left"
    )
    
    # 5. Remplir les valeurs NULL (cas où la jointure n'a pas trouvé de correspondance)
    enhanced_data = enhanced_data.fillna({
        "price_per_mile": 3.5,
        "payment_type_label": "credit_card",
        "is_weekend": 0
    })
    
    # 6. Calculer les prédictions du modèle original (formule de base)
    data_with_original = enhanced_data.withColumn(
        "original_prediction",
        F.lit(3.0) + 
        (F.col("trip_distance") * 2.5) + 
        (F.col("trip_duration_minutes") * 0.5) +
        F.when(F.col("time_of_day") == "evening", 2.0).otherwise(0) +
        F.when(F.col("is_airport_pickup") | F.col("is_airport_dropoff"), 5.0).otherwise(0) +
        (F.col("passenger_count") * 0.5)
    )
    
    # 7. Calculer les prédictions du modèle amélioré (ajustements basés sur les nouvelles features)
    data_with_both = data_with_original.withColumn(
        "improved_prediction",
        F.col("original_prediction") +
        
        # AMÉLIORATION 1: Ajustement selon le prix par mile
        F.when(F.col("price_per_mile") > 10, -5.0)
        .when(F.col("price_per_mile") < 2, 3.0)
        .otherwise(0) +
        
        # AMÉLIORATION 2: Ajustement selon le type de paiement
        F.when(F.col("payment_type_label") == "credit_card", 1.0)
        .when(F.col("payment_type_label") == "cash", -0.5)
        .otherwise(0) +
        
        # AMÉLIORATION 3: Ajustement week-end
        F.when(F.col("is_weekend") == 1, 2.5).otherwise(0)
    )
    
    # 8. Calculer les métriques pour les deux modèles
    metrics = data_with_both.agg(
        # Modèle original
        F.sqrt(F.avg((F.col("target_total_amount") - F.col("original_prediction")) ** 2)).alias("original_rmse"),
        F.avg(F.abs(F.col("target_total_amount") - F.col("original_prediction"))).alias("original_mae"),
        F.pow(F.corr("target_total_amount", "original_prediction"), 2).alias("original_r2"),
        
        # Modèle amélioré
        F.sqrt(F.avg((F.col("target_total_amount") - F.col("improved_prediction")) ** 2)).alias("improved_rmse"),
        F.avg(F.abs(F.col("target_total_amount") - F.col("improved_prediction"))).alias("improved_mae"),
        F.pow(F.corr("target_total_amount", "improved_prediction"), 2).alias("improved_r2")
    )
    
    # 9. Ajouter les métadonnées et les pourcentages d'amélioration
    result = metrics.select(
        "original_rmse",
        "original_mae",
        "original_r2",
        "improved_rmse",
        "improved_mae",
        "improved_r2",
        F.lit("2.0_with_3_features").alias("model_version"),
        F.current_timestamp().alias("training_date"),
        F.lit("price_per_mile, payment_type_label, is_weekend").alias("features_added"),
        F.round(((F.col("original_rmse") - F.col("improved_rmse")) / F.col("original_rmse")) * 100, 2).alias("rmse_improvement_pct"),
        F.round(((F.col("original_mae") - F.col("improved_mae")) / F.col("original_mae")) * 100, 2).alias("mae_improvement_pct")
    )
    
    return result