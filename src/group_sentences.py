import os
import pandas as pd

MAX_GROUP_TOKENS = 450


def rel_path(path, anchor=None):
    if anchor is None:
        anchor = __file__
    anchor_path = os.path.abspath(anchor)
    anchor_dir = os.path.dirname(anchor_path)
    return os.path.abspath(os.path.join(anchor_dir, path))


def group_sentences(sentences_df):
    sentences = sentences_df['sentence'].values
    num_tokens_all = sentences_df['num_bert_tokens'].values

    grouped = [sentences[0]]
    num_tokens = num_tokens_all[0] + 2

    for s, num_to_add in zip(sentences[1:], num_tokens_all[1:]):
        if num_tokens + num_to_add <= MAX_GROUP_TOKENS:
            num_tokens += num_to_add
            grouped[-1] = f'{grouped[-1]} {s}'
        else:
            num_tokens = num_to_add
            grouped.append(s)

    return grouped


df = pd.read_json(
    rel_path('../data/sentences.jsonl'),
    lines=True,
    orient='records',
)

gdf = df.groupby('article_id').apply(group_sentences)
sent_df = pd.DataFrame(gdf.values.tolist())
sent_df['article_id'] = gdf.index
sent_df = pd.melt(
    sent_df,
    id_vars='article_id',
    var_name='group_id',
    value_name='sentence',
)
sent_df.dropna(inplace=True)
sent_df.sort_values(['article_id', 'group_id'], inplace=True)
sent_df.to_json(
    rel_path('../data/sentences_grouped.jsonl'),
    lines=True,
    orient='records',
)
