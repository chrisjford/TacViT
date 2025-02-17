import config
from learning.train import train_model
from learning.test import evaluate_model
from ablation.run_ablation import run_ablation_study
from utils.logging_utils import setup_logging

if __name__ == "__main__":
    setup_logging()

    if config.ABLATION_STUDY:
        run_ablation_study()
    else:
        train_model()
        evaluate_model()
