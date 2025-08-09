class TrainingConfig:
    TRAIN_DIR = "data/train"
    VAL_DIR = "data/validation"
    MODEL_NAME = "naver-clova-ix/donut-base"
    # MODEL_NAME = "previous_trained_models/epoch_5"
    BATCH_SIZE = 2
    MAX_EPOCHS = 10
    # Not too high: prevents big jumps that could destabilize fine-tuning.
    # Not too low: still allows meaningful gradient updates.
    LEARNING_RATE = 4e-5
    CONFIDENCE_THRESHOLD = 0.8
    CHECKPOINT_DIR = "checkpoints"
    MAX_LENGTH = 128 
    