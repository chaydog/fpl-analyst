"""Run data pull and model training before gunicorn starts."""

import os
from pathlib import Path

os.chdir(Path(__file__).parent)

if not Path("data/processed/history.parquet").exists():
    print("=== Pulling FPL data ===")
    from fpl.ingest import DataIngestor
    ingestor = DataIngestor()
    ingestor.pull_all(verbose=True)

if not Path("models/models.pkl").exists():
    print("=== Training models ===")
    from fpl.features import FeatureBuilder
    from fpl.model import PointsPredictor
    fb = FeatureBuilder()
    fb.load_data()
    features = fb.build_training_features()
    feature_cols = fb.get_feature_columns()
    predictor = PointsPredictor()
    predictor.train(features, feature_cols)

print("=== Startup complete ===")
