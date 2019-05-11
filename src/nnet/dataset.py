import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer


def pad(token_ids, pad_value=0, expected_length=512):
    num_to_pad = expected_length - len(token_ids)
    return token_ids + [pad_value] * num_to_pad


class SentencesDataset(Dataset):
    def __init__(self, data, size=None):
        self.data = data
        self.size = size

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.pad_vid = tokenizer.vocab['[PAD]']

    def __len__(self):
        return self.size or len(self.data)

    def __getitem__(self, idx):
        # src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt
        input_ids, labels, segments_ids, cls_ids, _, _ = self.data[idx]

        input_t = torch.Tensor(pad(input_ids, pad_value=self.pad_vid)).long()
        segments_t = torch.Tensor(pad(segments_ids)).long()
        cls_t = torch.Tensor(pad(cls_ids, pad_value=-1)).long()
        labels_t = torch.Tensor(pad(labels)).float()
        attention_mask = 1 - (input_t == self.pad_vid)
        cls_mask = 1 - (cls_t == -1)
        return input_t, attention_mask, segments_t, cls_t, cls_mask, labels_t
