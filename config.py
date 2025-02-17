# Dataset Paths (External Storage)
DATASET_PATH = "/mnt/external_drive/tactile_dataset/master_labels.csv"

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = "cuda"

# Domain Adaptation Options
USE_MMD = False  # Disable MMD when using DANN
USE_DANN = True  # Enable DANN-based feature alignment
DANN_LAMBDA = 0.1  # Weight for domain classification loss
