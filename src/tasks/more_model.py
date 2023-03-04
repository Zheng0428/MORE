# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import transformers
from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU, MLPModel
import numpy as np

########################################
import torch
from torch.nn.parameter import Parameter
import math, time
########################################
# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20
HIDDEN_NUM = 10000
pad_id = 0
class MOREModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Build LXRT encoder
        self.more_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH,
            mode = 'l'
        )
#        hid_dim = self.more_encoder.dim
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2") #load tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        transformers.logging.set_verbosity_error()
        self.more_decoder = transformers.GPT2LMHeadModel.from_pretrained('gpt2', num_hidden_layers = 6)
        self.gpt_input_dim = self.more_decoder.lm_head.in_features
        self.gpt_output_dim = self.more_decoder.lm_head.out_features
        #build mlp model for mix sequence
        len_ia = self.more_encoder.max_seq_length
        len_is = 36                    #state's patch:36
        size = 19                      #output length : 19
        if size % 2 == 1:
            len_oa = size // 2
            len_os = size // 2 + 1
        else:
            len_os = len_oa = size // 2
        self.mlp_model_a = MLPModel(len_ia, HIDDEN_NUM, len_oa)
        self.mlp_model_s = MLPModel(len_is, HIDDEN_NUM, len_os)
        #resize target dim:
        # self.mlp_model_o = MLPModel(50257, num_oa)


    def calculate_loss_and_accuracy(self, outputs, labels, device):
        """
        计算非pad_id的平均loss和准确率
        :param outputs:
        :param labels:
        :param device:
        :return:
        """
        logits = outputs[0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
        # 用前n-1个token，预测出第n个token
        # 用第i个token的prediction_score用来预测第i+1个token。
        # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(device)

        loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),shift_labels.view(-1))

        _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

        # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
        not_ignore = shift_labels.ne(pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
        num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

        correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
        correct = correct.float().sum()

        accuracy = correct / num_targets
        loss = loss / num_targets
        return loss, accuracy

    def mlp(self, action, state, reward):
        action = self.mlp_model_a(action)
        state = self.mlp_model_s(state)
        reward = reward.unsqueeze(2).repeat(1, 1, 768)
        seq = torch.cat((action, state, reward),1)
        return seq


    def mix(self, action, state, reward):
        action = torch.flatten(action, 1, 2)
        state = torch.flatten(state, 1, 2)
        batch_size = action.shape[0]
        action = torch.cat([action, torch.ones(batch_size, 1).cuda()], dim=1)
        state = torch.cat([state, torch.ones(batch_size, 1).cuda()], dim=1)


        # 假设所设秩: R = 4, 期望融合后的特征维度: h = 768
        R, h = 4, self.gpt_input_dim - 1
        Wa = Parameter(torch.Tensor(R, action.shape[1], h)).cuda()
        Wa = torch.nn.init.xavier_normal_(Wa, gain=1.0).cuda()
        Ws = Parameter(torch.Tensor(R, state.shape[1], h)).cuda()
        Ws = torch.nn.init.xavier_normal_(Ws, gain=1.0).cuda()
        Wf = Parameter(torch.Tensor(1, R)).cuda()
        Wf = torch.nn.init.xavier_normal_(Wf, gain=1.0).cuda()
        bias = Parameter(torch.Tensor(1, h)).cuda()
        bias = torch.nn.init.xavier_normal_(bias, gain=1.0).cuda()

        # 分解后，并行提取各模态特征
        fusion_A = torch.matmul(action, Wa)
        fusion_S = torch.matmul(state, Ws)

        # 利用一个Linear再进行特征融合（融合R维度）
        funsion_AS = fusion_A * fusion_S
        funsion_AS = torch.matmul(Wf, funsion_AS.permute(1,0,2)).squeeze() + bias
        #fusion_AS = funsion_AS.detach().numpy()
        #最终输出的seq特征维度是（32，512）
        seq = torch.cat((funsion_AS, reward),1)
        return seq

    def forward(self, state, pos, action, reward, target):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param state: (b, o, f)
        :param pos:  (b, o, 4)
        :param action: (b,) Type -- list of string
        :param reward: (b, 1)
        :return: (b, 50257) 
        """
        texts = [
            'This is the first text.',
            'This is the second text.',
            'This is the third text.',
            'This is the fourth text.',
            'This is the fifth text.',
            'This is the sixth text.',
            'This is the seventh text.',
            'This is the eighth text.',
            'This is the ninth text.',
            'This is the tenth text.',
            'This is the eleventh text.',
            'This is the twelfth text.',
            'This is the thirteenth text.',
            'This is the fourteenth text.',
            'This is the fifteenth text.',
            'This is the sixteenth text.',
            'This is the seventeenth text.',
            'This is the eighteenth text.',
            'This is the nineteenth text.',
            'This is the twentieth text.',
            'This is the first text.',
            'This is the second text.',
            'This is the third text.',
            'This is the fourth text.',
            'This is the fifth text.',
            'This is the sixth text.',
            'This is the seventh text.',
            'This is the eighth text.',
            'This is the ninth text.',
            'This is the tenth text.',
            'This is the eleventh text.',
            'This is the twelfth text.',
        ]
        x = self.more_encoder(action, (state, pos))
        seq = self.mlp(x[0], x[1], reward)   #mix the output of the lxmert and the reward (numpy)
        target = self.tokenizer.batch_encode_plus(texts, padding = 'max_length', truncation = True, return_tensors = 'pt', max_length = MAX_VQA_LENGTH, add_special_tokens = True, return_attention_mask = True, return_token_type_ids = False)
        output = self.more_decoder(inputs_embeds = seq, labels = target.data['input_ids'].cuda())        #Be sure to pay attention to whether the input sequences are of the same length  #past_key_values = past 后面有时间可以加上
        loss, accuray = self.calculate_loss_and_accuracy(outputs = output.logits, labels = action.data['input_ids'], device = 'cuda')
        return output


