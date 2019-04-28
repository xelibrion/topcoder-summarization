import os
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
from multiprocessing import Pool

BERT_MODEL = 'bert-base-uncased'


def rel_path(path, anchor=None):
    if anchor is None:
        anchor = __file__
    anchor_path = os.path.abspath(anchor)
    anchor_dir = os.path.dirname(anchor_path)
    return os.path.abspath(os.path.join(anchor_dir, path))


df = pd.read_json(
    rel_path('../data/unpacked.jsonl'),
    lines=True,
    orient='records',
)


def initializer():
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)


def tokenize(input_tuple):
    idx, abstract, article = input_tuple
    return idx, tokenizer.tokenize(abstract), tokenizer.tokenize(article)


df = df
abstracts = df['abstract']
articles = df['article']
total_items = df.shape[0]

all_tokens = []

with tqdm(total=total_items) as tq:
    with Pool(12, initializer=initializer) as pool:

        args = list(zip(range(total_items), abstracts, articles))
        for t in pool.imap(tokenize, args):
            tq.update()
            all_tokens.append(t)

        pool.close()
        pool.join()

df_result = pd.DataFrame(all_tokens,
                         columns=[
                             'article_id',
                             'abstract_tokens',
                             'article_tokens',
                         ])
df_result.to_json(
    rel_path('../data/train_tokens.jsonl'),
    lines=True,
    orient='records',
)
