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
        self.gpt_input_dim = self.more_decoder.lm_head.in_features
        self.gpt_output_dim = self.more_decoder.lm_head.out_features
        #build mlp model for mix sequence
        num_ia = self.more_encoder.max_seq_length * self.more_encoder.dim
        num_is = 36 * self.more_encoder.dim  #state's patch:36
        size = self.gpt_input_dim - 1  #GPT-2 embed input dim:768,our for reward
        if size % 2 == 1:
            num_oa = size // 2
            num_os = size // 2 + 1
        else:
            num_os = num_oa = size // 2
        def hidden_size(input, output):
            return (input + output) * 2 // 3
        a = time.time()
        self.mlp_model_a = MLPModel(num_ia, hidden_size(num_ia, num_oa), num_oa)
        b = time.time()
        self.mlp_model_s = MLPModel(num_is, hidden_size(num_is, num_os), num_os)
        c = time.time()
        #resize target dim:
        self.mlp_model_o = MLPModel(self.gpt_output_dim, hidden_size(self.gpt_output_dim, num_oa), num_oa)
        d = time.time()
        print (b-a)
        print (c-b)
        print (d-c)


    def mlp(self, action, state, reward):
        action = self.mlp_model_a(torch.flatten(action, 1, 2))
        state = self.mlp_model_s(torch.flatten(state, 1, 2))
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
        seq = self.mix(x[0], x[1], reward)   #mix the output of the lxmert and the reward (numpy)
        output = self.more_decoder(inputs_embeds = seq)  #past_key_values = past 后面有时间可以加上
        tmp = torch.flatten(x[0], 1, 2)
        target = self.mlp_model_o(torch.flatten(x[0], 1, 2))
        return output ,target


