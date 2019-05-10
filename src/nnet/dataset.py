import torch
from torch.utils.data import Dataset


def pad(token_ids, expected_length=512):
    to_pad = expected_length - len(token_ids)
    return token_ids + [0] * to_pad


class SentencesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt
        input_ids, labels, segments_ids, cls_ids, _, _ = self.data[idx]

        input_t = torch.Tensor(pad(input_ids)).long()
        segments_t = torch.Tensor(pad(segments_ids)).long()
        cls_t = torch.Tensor(pad(cls_ids)).long()
        labels_t = torch.Tensor(pad(labels)).float()
        attention_mask = 1 - (input_t == 0)
        cls_mask = 1 - (cls_t == 0)
        return input_t, attention_mask, segments_t, cls_t, cls_mask, labels_t
