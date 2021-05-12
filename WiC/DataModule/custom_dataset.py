import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class BertDataset(Dataset):
    def __init__(self, sentences, word_ranges, labels, max_tokens, tokenizer=None):
        self.sentences = sentences
        self.word_ranges = word_ranges
        self.labels = labels
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")
        self.max_tokens = max_tokens

        self.INDICES_PADDING_VALUE = 5
        self.MAX_TOKENS = 100
        self.INDICES_PADDING_LEN = 5

    def _tokenize(self, sentence):
        return self.tokenizer(sentence,
                              add_special_tokens=True,
                              max_length=self.max_tokens,
                              padding="max_length",
                              truncation=True,
                              return_offsets_mapping=True)

    def _get_input_ids_indices_for_word(self, offset_mapping, word_start, word_end):
        indices = []
        for idx, (start, end) in enumerate(offset_mapping):
            if start != end and word_start <= start and end <= word_end:
                indices.append(idx)
            elif word_start < start:
                break

        indices.extend([self.INDICES_PADDING_VALUE for i in range(self.INDICES_PADDING_LEN - len(indices))])
        return torch.tensor(indices)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        first_sentence, second_sentence = self.sentences[index]
        (first_word_start, first_word_end), (second_word_start, second_word_end) = self.word_ranges[index]

        first_input = self._tokenize(first_sentence)
        second_input = self._tokenize(second_sentence)

        input_ids = (torch.tensor(first_input["input_ids"]), torch.tensor(second_input["input_ids"]))
        attention_masks = (torch.tensor(first_input["attention_mask"]), torch.tensor(second_input["attention_mask"]))

        first_word_ids_indices = self._get_input_ids_indices_for_word(first_input["offset_mapping"], first_word_start,
                                                                      first_word_end)
        second_word_ids_indices = self._get_input_ids_indices_for_word(second_input["offset_mapping"],
                                                                       second_word_start, second_word_end)

        word_ids_indices = (first_word_ids_indices, second_word_ids_indices)

        return input_ids, attention_masks, word_ids_indices, torch.tensor(self.labels[index], dtype=torch.float)
