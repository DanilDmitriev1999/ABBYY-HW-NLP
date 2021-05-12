import json
import pandas as pd

from sklearn.model_selection import train_test_split


def read_train_data(path, read_tags=False):
    with open(path) as f:
        df = pd.DataFrame(json.load(f))

    if read_tags:
        if 'test' in path:
            tags_path = 'data/MCL-WiC/test/test.en-en.gold'
        else:
            tags_path = path[:path.find('.data')] + '.gold'
        with open(tags_path) as f:
            df = df.merge(pd.DataFrame(json.load(f)))
        df['tag'] = df['tag'].replace({'T': 1, 'F': 0})

    df['lemma'] = df['lemma'].apply(lambda lemma: lemma.lower())
    return df


def lemma_train_test_split(df, test_size=0.2):
    unique_lemmas = sorted(set(df['lemma'].tolist()))
    train_lemmas, test_lemmas = train_test_split(unique_lemmas, test_size=test_size, random_state=1)
    df_train = df[df['lemma'].isin(train_lemmas)]
    df_test = df[df['lemma'].isin(test_lemmas)]
    return df_train, df_test

if __name__ == '__main__':
    pass