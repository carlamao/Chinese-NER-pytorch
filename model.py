import torch.nn as nn
# from .transformers.modeling_bert import BertPreTrainedModel
# from .transformers.modeling_bert import BertModel
from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss
# from losses.focal_loss import FocalLoss
# from losses.label_smoothing import LabelSmoothingCrossEntropy
from CRF import CRF


class BertSoftmaxForNer(BertPreTrainedModel):
    # def __init__(self, config, num_labels):
    #     super(BertSoftmaxForNer, self).__init__(config)
    #     self.num_labels = num_labels
    #     self.bert = BertModel(config)
    #     self.dropout = nn.Dropout(config.hidden_dropout_prob)
    #     self.classifier = nn.Linear(config.hidden_size, num_labels)
    #     self.apply(self.init_bert_weights)
    #
    # def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
    #     _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
    #     pooled_output = self.dropout(pooled_output)
    #     logits = self.classifier(pooled_output)
    #
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()
    #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    #         return loss
    #     else:
    #         return logits
    def __init__(self, config):
        super(BertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #print("number_labels: ", config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.loss_type = config.loss_type
        #print(config.loss_type)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        #print("bert output: ", sequence_output.size())
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        #print("logits: ", logits.size())
        #outputs = logits
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            #print("labels: ", labels.size())
            # assert self.loss_type in ['lsr', 'focal', 'ce']
            # if self.loss_type == 'lsr':
            #     loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            # elif self.loss_type == 'focal':
            #     loss_fct = FocalLoss(ignore_index=0)
            # else:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_logits = logits.view(-1, self.num_labels)[active_loss]
            #     active_labels = labels.view(-1)[active_loss]
            #     loss = loss_fct(active_logits, active_labels)
            # else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None,input_lens=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores