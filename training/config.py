class TrainingConfig:
    TRAIN_DIR = "data/train"
    VAL_DIR = "data/validation"
    MODEL_NAME = "naver-clova-ix/donut-base"
    BATCH_SIZE = 2
    MAX_EPOCHS = 5
    LEARNING_RATE = 5e-5
    CONFIDENCE_THRESHOLD = 0.8
    CHECKPOINT_DIR = "checkpoints"
    