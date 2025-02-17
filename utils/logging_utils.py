import os
import logging

def setup_logging():
    log_dir = "experiments/logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "experiment.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Logging initialized.")

def log_experiment(epoch, loss):
    logging.info(f"Epoch {epoch+1}, Loss: {loss:.4f}")
