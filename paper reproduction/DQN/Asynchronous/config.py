NUM_PS = 1
NUM_WORKER = 1
USE_GPU = False
CUDA_VISIBLE_DEVICES = "0"
PER_PROCESS_GPU_MEMORY_FRACTION = 1 / ((1 + 0.1 * (NUM_PS + NUM_WORKER)) * (NUM_PS + NUM_WORKER))

LEARNING_RATE = [1e-4, 1e-5]
LR_ANNEAL_STEP = [400000] # 4e5

ENV_NAME = "PongNoFrameskip-v4"
TOTAL_STEP = 500000 # 5e5
BATCH_SIZE = 32
BUFFER_SIZE = 50000 // NUM_WORKER
INITIAL_BUFFER_SIZE = 10000 // NUM_WORKER
EPSILON_MAX = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_STEP = 250000 # 2.5e5
REWARD_DISCOUNT = 0.99
TARGET_NETWORK_UPDATE_STEP = 1000
AUTOSAVE_STEP = 10000

SAVE_DIR = "./Saves/"
FIGURE_TRAINING_DIR = "./Figures/Training/"
FIGURE_VISUALIZATION_DIR = "./Figures/Visualization/"