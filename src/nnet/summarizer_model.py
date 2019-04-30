import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertModel


class Summarizer(nn.Module):
    def __init__(self):
        super(Summarizer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.decoder = Classifier(self.bert.config.hidden_size)

        for p in self.decoder.parameters():
            nn.init.uniform_(p, -0.1, 0.1)

    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, input_ids, attention_mask, segments, cls_ids, cls_mask):
        encoded_input, _ = self.bert(
            input_ids,
            token_type_ids=segments,
            attention_mask=attention_mask,
            output_all_encoded_layers=False,
        )
        sents_vec = encoded_input[torch.arange(encoded_input.size(0)).
                                  unsqueeze(1), cls_ids]
        sents_vec = sents_vec * cls_mask[:, :, None].float()
        sent_scores = self.decoder(sents_vec, cls_mask).squeeze(-1)
        return sent_scores, cls_mask


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores
