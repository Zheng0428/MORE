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
########################################
# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


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
        transformers.logging.set_verbosity_error()
        self.more_decoder = transformers.GPT2LMHeadModel.from_pretrained('gpt2', num_hidden_layers = 6)
        # # VQA Answer heads
        # self.logit_fc = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim * 2),
        #     GeLU(),
        #     BertLayerNorm(hid_dim * 2, eps=1e-12),
        #     nn.Linear(hid_dim * 2, num_answers)
        # )
        # self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def mlp(self, action, state, reward, size):
        action, state = action.cpu(), state.cpu()
        num_ia = action.shape[1] * action.shape[2]
        num_is = state.shape[1] * state.shape[2]
        num_h = 10000
        if size % 2 == 1:
            num_oa = size // 2
            num_os = size // 2 + 1
        else:
            num_os = num_oa = size // 2
        mlp_model_a= MLPModel(num_ia, num_h, num_oa)
        mlp_model_s= MLPModel(num_is, num_h, num_os)
        action = mlp_model_a(torch.flatten(action, 1, 2))
        state = mlp_model_s(torch.flatten(state, 1, 2))
        seq = np.concatenate((action.detach().numpy(), state.detach().numpy(), reward.detach().numpy()),1)
        return seq


    def mix(self, action, state, reward, size):
        action = torch.flatten(action, 1, 2).cpu()
        state = torch.flatten(state, 1, 2).cpu()

        batch_size = action.shape[0]
        action = torch.cat([action, torch.ones(batch_size, 1)], dim=1)
        state = torch.cat([state, torch.ones(batch_size, 1)], dim=1)


        # 假设所设秩: R = 4, 期望融合后的特征维度: h = 768
        R, h = 4, size
        Wa = Parameter(torch.Tensor(R, action.shape[1], h))
        Wa = torch.nn.init.xavier_normal_(Wa, gain=1.0)
        Ws = Parameter(torch.Tensor(R, state.shape[1], h))
        Ws = torch.nn.init.xavier_normal_(Ws, gain=1.0)
        Wf = Parameter(torch.Tensor(1, R))
        Wf = torch.nn.init.xavier_normal_(Wf, gain=1.0)
        bias = Parameter(torch.Tensor(1, h))
        bias = torch.nn.init.xavier_normal_(bias, gain=1.0)

        # 分解后，并行提取各模态特征
        fusion_A = torch.matmul(action, Wa)
        fusion_S = torch.matmul(state, Ws)

        # 利用一个Linear再进行特征融合（融合R维度）
        funsion_AS = fusion_A * fusion_S
        funsion_AS = torch.matmul(Wf, funsion_AS.permute(1,0,2)).squeeze() + bias
        #fusion_AS = funsion_AS.detach().numpy()
        #最终输出的seq特征维度是（32，512）
        seq = np.concatenate((funsion_AS.detach().numpy(), reward.detach().numpy()),1)
        return seq

    def forward(self, state, pos, action, reward):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param state: (b, o, f)
        :param pos:  (b, o, 4)
        :param action: (b,) Type -- list of string
        :param reward: (b, 1)
        :return: (b, 50257) 
        """
        x = self.more_encoder(action, (state, pos))
        #reward = torch.round(torch.rand(32, 1))   #test
        seq = self.mlp(x[0], x[1], reward, 767)   #mix the output of the lxmert and the reward (numpy)
        output = self.more_decoder(inputs_embeds = torch.from_numpy(seq).cuda())  #past_key_values = past 后面有时间可以加上
        return output


