from models.sent_eval import ELMoEmbeddingEvaluator
import numpy as np


class GetELMoVector():
    
    def get_elmo_vector(self,sentence):

        model = ELMoEmbeddingEvaluator(tune_model_fname="data/sentence-embeddings/elmo/tune-ckpt",
                                        pretrain_model_fname="data/sentence-embeddings/elmo/pretrain-ckpt/elmo.model",
                                        options_fname="data/sentence-embeddings/elmo/pretrain-ckpt/options.json",
                                        vocab_fname="data/sentence-embeddings/elmo/pretrain-ckpt/elmo-vocab.txt",
                                        max_characters_per_token=30, dimension=256, num_labels=2)


        result = model.get_token_vector_sequence(sentence)
        array = np.zeros((1,256,256))

        index1 = 0
        for i in result[1]:
            array[0][index1] = i
            index1 += 1

        print(array)

        return array

