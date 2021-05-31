
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import seq2seq.common.util as util
import numpy as np
from typing import *
import logging

from seq2seq.common.util_checkpoint import load_ckpt

# k_model_name = 't5-small'
# k_path = '/Users/jsrozner/jsrozner/cryptic/cryptic-data/cluster_save/t5-small-json-128b-run-20210318_133500-3gxvrh70/epoch_13.pth.tar'

class ModelRunner:
    def __init__(self, model_name, ckpt_path,
                 num_generations=10):
        device, gpu_ids = util.get_available_devices(assert_cuda=True)
        logging.info(device, gpu_ids)

        self.device = device
        self.num_generations = num_generations
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)

        load_ckpt(ckpt_path, self.model, map_location=device)
        self.model.to(device)

    def generate(self, sentences: List[str]) -> List[np.array]:
        # tokenized = self.tokenizer.batch_encode_plus(sentences, max_length=50, return_tensors='pt', padding='max_length', truncation=True)
        tokenized = self.tokenizer(sentences, padding='longest', return_tensors='pt')
        input_ids = tokenized['input_ids'].to(self.device)
        src_mask = tokenized['attention_mask'].to(self.device)

        # greedy decoding
        # out_ids = self.model.generate(input_ids, attention_mask=src_mask)
        # greedy_decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        generated_ids_sampled = self.model.generate(input_ids,
                                                    attention_mask=src_mask,
                                                    num_beams=self.num_generations,
                                                    num_return_sequences=self.num_generations,
                                                    do_sample=False,
                                                    max_length=10,
                                                    length_penalty=0.05
                                                    )

        decoded = self.tokenizer.batch_decode(generated_ids_sampled, skip_special_tokens=True)
        decoded = np.array_split(decoded, len(sentences))

        return decoded

        # for output, input_sent in zip(decoded, sentences):
        #     print(input_sent)
        #     print(output)

        # set comparison
        # orig_set, other_sets = decoded_sets[0], decoded_sets[1:]
        # decoded_sets = list(map(set, decoded))
        # for new_set, sent in zip(other_sets, sentences[1:]):
        #     print(sent)
        #     pp(new_set)
        #     pp(f'new: {new_set.difference(orig_set)}')
        #     pp(f'lost: {orig_set.difference(new_set)}')
        #     print()

