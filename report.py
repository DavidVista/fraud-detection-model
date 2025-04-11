from detection_pipeline import ModelPipeline


if __name__ == "__main__":
    pipeline = ModelPipeline(
        data_path='encoded_fraud_data.parquet',
        sample_frac=0.5
    )

    # Feature selection
    print("Starting Genetic Algorithm for feature selection...")
    features = pipeline.genetic_feature_selection()
    print(features)
    with open('features_selected.txt', 'w') as f:
        f.write(str(features))

    results = pipeline.run_pipeline(
        models_to_run=[
            'RandomForest',
            'XGBoost',
            'LightGLM',
            'DenseNN',
            'RNN'
        ]
    )
