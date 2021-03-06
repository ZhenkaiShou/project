UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"

SOURCE_TRAINING_FILE = "train.en"
TARGET_TRAINING_FILE = "train.vi"
SOURCE_TEST_FILE_1 = "tst2012.en"
TARGET_TEST_FILE_1 = "tst2012.vi"
SOURCE_TEST_FILE_2 = "tst2013.en"
TARGET_TEST_FILE_2 = "tst2013.vi"
SOURCE_VOCAB_FILE = "vocab.en"
TARGET_VOCAB_FILE = "vocab.vi"

MAX_SEQUENCE_LENGTH = 200
EMBEDDING_UNIT = 128
HIDDEN_UNIT = 128
RNN_LAYER = 2
DROPOUT = 0.2
BUCKET_DECAY = 0.7
BUCKET_NUM = 8
BUCKET_BATCH_SIZE = 50
LEARNING_RATE = 3e-4
TRAINING_STEP = 60000
BEAM_WIDTH = 10
COVERAGE_PENALTY = 0.5

DATA_DIR = "./Data/"
SAVE_DIR = "./Saves/"