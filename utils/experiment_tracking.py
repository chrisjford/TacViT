import os
import json
import logging

def setup_experiment_logging():
    """
    Initializes logging for experiment tracking.
    """
    log_dir = "experiments/logs"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_dir, "experiment_tracking.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def log_experiment_results(experiment_name, config, test_loss):
    """
    Logs experiment results in JSON format.

    Args:
        experiment_name (str): Name of the experiment (e.g., "DANN", "MMD", "No DA").
        config (dict): Hyperparameter settings used for the experiment.
        test_loss (float): Final test loss of the model.
    """
    log_path = "experiments/logs/experiment_results.json"
    
    # Load existing logs if they exist
    if os.path.exists(log_path):
        with open(log_path, "r") as file:
            results = json.load(file)
    else:
        results = {}

    # Store results
    results[experiment_name] = {
        "test_loss": test_loss,
        "config": config
    }

    # Save updated logs
    with open(log_path, "w") as file:
        json.dump(results, file, indent=4)

    logging.info(f"Experiment: {experiment_name}, Test Loss: {test_loss:.4f}")
