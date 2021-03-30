from models.sent_eval import ELMoEmbeddingEvaluator
import csv
import numpy as np

test_path = "elmo/vec_test_x.csv"
train_path = "elmo/vec_train_x.csv"

f_test = open(test_path, 'r')
r_test = csv.reader(f_test)

f_train = open(train_path, 'r')
r_train = csv.reader(f_train)

	


model = ELMoEmbeddingEvaluator(tune_model_fname="data/sentence-embeddings/elmo/tune-ckpt",
                                 pretrain_model_fname="data/sentence-embeddings/elmo/pretrain-ckpt/elmo.model",
                                 options_fname="data/sentence-embeddings/elmo/pretrain-ckpt/options.json",
                                 vocab_fname="data/sentence-embeddings/elmo/pretrain-ckpt/elmo-vocab.txt",
                                 max_characters_per_token=30, dimension=256, num_labels=2)

test_arr = []
for line in r_train:
	result = model.get_token_vector_sequence(line[0])
	test_arr.append(result[1])


result_arr = np.array(test_arr)
np.save("elmo/raw_vector_train", result_arr)
