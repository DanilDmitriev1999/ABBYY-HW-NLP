import torch
import numpy as np
from torchtext.data import Field, Example, Dataset, BucketIterator
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PretrainingData:
    def __init__(self, file, DEVICE, batch_first=False, batch_size=64):
        assert type(batch_first) == bool
        assert type(file) == str
        assert file[-3:] == 'txt'
        
        self.batch_size = batch_size
        self.file = file
        
        self.MAX_TOKENS_COUNT = 16
        self.SUBSET_SIZE = 0.3
        self.BOS_TOKEN = '<s>'
        self.EOS_TOKEN = '</s>'
        
        self.source_field = Field(tokenize='spacy', init_token=BOS_TOKEN,
                                   eos_token=EOS_TOKEN, lower=True,
                                    batch_first=batch_first)
        
        self.target_field = Field(tokenize='moses', init_token=BOS_TOKEN,
                                   eos_token=EOS_TOKEN,lower=True,
                                    batch_first=batch_first)
        
        self.fields = [('source', self.source_field), ('target', self.target_field)]
        
    def read_data(self):
        
        self.examples = []
        with open(self.file) as f:
            for line in tqdm(f, total=328190):
                source_text, target_text, _ = line.split('\t')
                source_text = self.source_field.preprocess(source_text)
                target_text = self.target_field.preprocess(target_text)
                if len(source_text) <= self.MAX_TOKENS_COUNT and len(target_text) <= self.MAX_TOKENS_COUNT:
                    if np.random.rand() < self.SUBSET_SIZE:
                        self.examples.append(Example.fromlist([source_text, target_text], self.fields))
                        
    def start(self):
        read_data()
        dataset = Dataset(self.examples, self.fields)
        train_dataset, test_dataset = dataset.split(split_ratio=0.85)
        print('Train size =', len(train_dataset))
        print('Test size =', len(test_dataset))
        source_field.build_vocab(train_dataset, min_freq=2)
        print('Source vocab size =', len(source_field.vocab))
        target_field.build_vocab(train_dataset, min_freq=2)
        print('Target vocab size =', len(target_field.vocab))
        
        train_iter, test_iter = BucketIterator.splits(
            datasets=(train_dataset, test_dataset),
            batch_sizes=(self.batch_size, self.batch_size),
            sort_within_batch = True,
            sort_key = lambda x : len(x.source),
            device=DEVICE,
        )
        return train_iter, test_iter, self.source_field, self.target_field