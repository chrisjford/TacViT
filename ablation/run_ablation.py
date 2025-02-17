from ablation.variants import get_ablation_models
from learning.train import train_model
from learning.test import evaluate_model
import logging

def run_ablation_study():
    """
    Runs an ablation study by training and evaluating different model variants.
    Logs the results for comparison.
    """
    models = get_ablation_models()
    results = {}

    logging.info("Starting Ablation Study")

    for name, model in models.items():
        logging.info(f"Training ablation variant: {name}")
        
        # Train model variant
        train_model(model)
        
        # Evaluate model variant
        test_loss = evaluate_model(model)
        results[name] = test_loss

        logging.info(f"Ablation Variant: {name}, Test Loss: {test_loss:.4f}")

    logging.info("Ablation Study Completed")
    print("Ablation Study Results:", results)
