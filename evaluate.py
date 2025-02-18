import config
from learning.train import train_model
from learning.test import evaluate_model
from ablation.run_ablation import run_ablation_study
from utils.logging_utils import setup_logging

evaluate_model()
