import os
from urllib.request import urlretrieve

from config import *

def download_file(file_url, file_name):
  if not os.path.isfile(file_name):
    urlretrieve(file_url, file_name)

def download():
  if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)
  
  site_url = "https://nlp.stanford.edu/projects/nmt/data/"
  # Download the dataset if not exist.
  download_file(site_url + "iwslt15.en-vi/train.en", DATA_DIR + SOURCE_TRAINING_FILE)  
  download_file(site_url + "iwslt15.en-vi/train.vi", DATA_DIR + TARGET_TRAINING_FILE)
  
  download_file(site_url + "iwslt15.en-vi/tst2012.en", DATA_DIR + SOURCE_TEST_FILE_1)
  download_file(site_url + "iwslt15.en-vi/tst2012.vi", DATA_DIR + TARGET_TEST_FILE_1)
  
  download_file(site_url + "iwslt15.en-vi/tst2013.en", DATA_DIR + SOURCE_TEST_FILE_2)
  download_file(site_url + "iwslt15.en-vi/tst2013.vi", DATA_DIR + TARGET_TEST_FILE_2)
  
  download_file(site_url + "iwslt15.en-vi/vocab.en", DATA_DIR + SOURCE_VOCAB_FILE)
  download_file(site_url + "iwslt15.en-vi/vocab.vi", DATA_DIR + TARGET_VOCAB_FILE)