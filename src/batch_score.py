"""Nightly batch scoring entry point (for cron / Task Scheduler).

    python -m src.batch_score
"""
from src import pipeline

if __name__ == "__main__":
    engine = pipeline.init_and_seed()
    result = pipeline.refresh_features_and_predictions(engine, pipeline.get_service(engine))
    print("Batch scoring complete:", result)
