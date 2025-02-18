# Dataset Paths (External Storage)
DATASET_PATH = "F:/TacViT/data/master_labels.csv"
TEST_SENSOR_ID = 2

# Training Parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 50
DEVICE = "cuda"
EARLY_STOPPING_PATIENCE = 10

ABLATION_STUDY = False

# Domain Adaptation Options
NUM_DOMAINS = 4
USE_TRADABOOST = True
TRADABOOST_ITERS = 10
USE_MMD = True  # Disable MMD when using DANN
USE_DANN = False  # Enable DANN-based feature alignment
DANN_LAMBDA = 0.1  # Weight for domain classification loss
MMD_LAMBDA = 0.1
TRADABOOST_LAMBDA = 0.1
