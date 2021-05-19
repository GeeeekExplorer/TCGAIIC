from transformers import BertPreTrainedModel, BertModel
import torch
from torch import nn
import config


class GAIICBert(BertPreTrainedModel):
    def __init__(self, bertconfig):
        super().__init__(bertconfig)
        self.num_labels = bertconfig.num_labels
        self.attack = False
        self.input = {}
        self.embdata = None
        self.embgrad = None

        self.bert = BertModel(bertconfig)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.classifier = nn.Linear(config.num_hidden_states, bertconfig.num_labels)

        self.init_weights()

        def forward_pre_hook(module, input):
            if self.training and not self.attack:
                self.embdata = module.weight.data.clone()
            if self.training and self.attack:
                norm = torch.norm(self.embgrad)
                if norm != 0 and not torch.isnan(norm):
                    module.weight.data.add_(1. * self.embgrad / norm)
        self.bert.embeddings.word_embeddings.register_forward_pre_hook(forward_pre_hook)

        def backward_hook(module, grad_input, grad_output):
            if self.training and not self.attack:
                self.embgrad = grad_input[0].clone()
                self.attack = True
                loss, _ = self.forward(**self.input)
                loss.requires_grad_(True)
                loss.backward()
                module.weight.data = self.embdata
                self.attack = False
        self.bert.embeddings.word_embeddings.register_backward_hook(backward_hook)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if self.training and not self.attack:
            self.input['input_ids'] = input_ids
            self.input['attention_mask'] = attention_mask
            self.input['position_ids'] = position_ids
            self.input['labels'] = labels

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
