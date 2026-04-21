import math
from typing import Callable, Optional
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Pscan import pscan
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoConfig, AutoModel, AutoTokenizer


class InMambaEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        
        # if self.c_in<=self.d_model and self.c_in <= d_compress_max:
        #     d_compress = None
        # else:
        #     d_compress = min(self.d_model, d_compress_max)
        self.layers = nn.ModuleList([InMambaBlock(configs) for _ in range(configs.e_layers)])

    def forward(self, x):
        # x : [bs * nvars, patch_num, d_model]

        for layer in self.layers:
            x = layer(x)

        x = F.silu(x)

        return x


class InMambaBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.gala = MambaBlock(configs)

        self.dropout = nn.Dropout(configs.dropout)
        self.configs = configs

    def forward(self, x):
        # x : [bs * nvars, patch_num, d_model]

        # output : [bs * nvars, patch_num, d_model]

        output = self.gala(x) 
        output = self.dropout(output)
        # output += x
        return output
  
class MambaBlock(nn.Module):
    """
    MambaModule, similar to https://arxiv.org/pdf/2402.18959
    """
    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(configs.d_model, 2 * configs.d_ff, bias=configs.bias)
        
        # projects x to input-dependent Δ, B, C, D
        self.x_proj = nn.Linear(configs.d_ff, configs.dt_rank + 2 * configs.d_state + configs.d_ff, bias=False)

        # projects Δ from dt_rank to d_ff
        self.dt_proj = nn.Linear(configs.dt_rank, configs.d_ff, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = configs.dt_rank**-0.5 * configs.dt_scale
        if configs.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif configs.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # dt bias
        dt = torch.exp(
            torch.rand(configs.d_ff) * (math.log(configs.dt_max) - math.log(configs.dt_min)) + math.log(configs.dt_min)
        ).clamp(min=configs.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = torch.arange(1, configs.d_state + 1, dtype=torch.float32).unsqueeze(0)

        self.A_log = nn.Parameter(torch.log(A))


        # 外部的降维后再计算
        self.d_inner = configs.d_ff // 2
        self.exit_proj_x = nn.Linear(configs.d_ff, self.d_inner)
        self.exit_proj_z = nn.Linear(configs.d_ff, self.d_inner)

        # SSM_exit的维度需要将ssm的d_ff变成d_inner     
        # projects x to input-dependent Δ, B, C, D
        self.x_proj_exit = nn.Linear(self.d_inner, configs.dt_rank + 2 * configs.d_state + self.d_inner, bias=False)

        # projects Δ from dt_rank to d_ff
        self.dt_proj_exit = nn.Linear(configs.dt_rank, self.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = configs.dt_rank**-0.5 * configs.dt_scale
        if configs.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif configs.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # dt bias
        dt_exit = torch.exp(
            torch.rand(self.d_inner) * (math.log(configs.dt_max) - math.log(configs.dt_min)) + math.log(configs.dt_min)
        ).clamp(min=configs.dt_init_floor)
        inv_dt_exit = dt_exit + torch.log(-torch.expm1(-dt_exit)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj_exit.bias.copy_(inv_dt_exit)

        # S4D real initialization
        A_exit = torch.arange(1, configs.d_state + 1, dtype=torch.float32).unsqueeze(0)

        self.A_log_exit = nn.Parameter(torch.log(A_exit))

        # projects block output from ED back to D
        self.out_inner_proj = nn.Linear(2*configs.d_ff, configs.d_ff, bias=configs.bias)
        self.out_exit_proj = nn.Linear(2*self.d_inner, self.d_inner, bias=configs.bias)
        self.up_dim = nn.Linear(self.d_inner, configs.d_ff, bias=configs.bias)

        # LLM
        
        # self.llm_input = nn.Linear(2*configs.d_ff, configs.llm_dim)
        # self.qwen2_config = AutoConfig.from_pretrained('/remote-home/share/dmb_nas/jxd/language_model/DeepSeek-R1-Distill-Qwen-1.5B')
        # # self.qwen2_config = AutoConfig.from_pretrained('/remote-home/share/dmb_nas2/wangjiaqi/exp2/llm-path/deepseek7b')
        # self.qwen2_config.num_hidden_layers = configs.llm_layers  # 设置隐藏层数
        # # self.qwen2_config.output_attentions = True
        # self.qwen2_config.output_hidden_states = True
        # self.llm_model = AutoModel.from_pretrained(
        #     '/remote-home/share/dmb_nas/jxd/language_model/DeepSeek-R1-Distill-Qwen-1.5B',
        #     trust_remote_code=True,
        #     local_files_only=True,
        #     config=self.qwen2_config,
        # )

        self.df_proj = nn.Linear(configs.d_model,configs.d_ff)
        self.Recurrent_proj = nn.Linear(2*configs.d_ff, configs.d_ff)
        self.inout_proj = nn.Linear(2*configs.d_ff, configs.d_ff)
        # Conv
        self.conv_size = configs.dconv

        # Final Proj
        self.out_proj = nn.Linear(2*configs.d_ff, configs.d_model)
        # self.out_proj = nn.Linear(configs.llm_dim, configs.d_model)
        self.hidden_dff = nn.Linear(2*configs.d_ff, configs.d_ff)
        self.hidden_dinner = nn.Linear(2*configs.d_ff, self.d_inner)

        self.queryinner_proj = nn.Linear(configs.d_ff, configs.d_state)
        self.keyinner_proj = nn.Linear(configs.d_state, configs.d_state)

        self.queryExternal_proj = nn.Linear(self.d_inner, configs.d_state)
        self.keyExternal_proj = nn.Linear(configs.d_state, configs.d_state)

        self.activate = nn.Softplus()
        # self.activate = nn.SiLU() #nn.ELU() #nn.SiLU()
        


    def forward(self, x):
        # x : [bs , seq_len, d_model]
        
        # y : [bs , seq_len, d_model]

        _, L, _ = x.shape
        
        # d_ff 设为 4 倍 d_model d_inner 设为 2倍 d_model
        hidden_states = self.in_proj(x) # [bs , seq_len, 2 * d_ff]

        hidden_states_fwd = self.hidden_dff(hidden_states) # [bs , seq_len, d_ff]
        hidden_states_bwd = self.hidden_dinner(hidden_states.flip([1])) # [bs , seq_len, d_inner]

        # recurrent_states = self.Recurrent_part(self.df_proj(x)) # [bs , seq_len , d_model]

        x, z = hidden_states.chunk(2, dim=-1) # [bs , seq_len , d_ff], [bs , seq_len , d_ff]

        # two stage branch
        # x branch
        x = F.silu(x) # [bs , seq_len , d_ff]   方向 ->
        x_filp = F.silu(x.flip([1])) # [bs , seq_len , d_inner]    方向 <-
        # x branch conv
        # x_conv, _ = self.conv1d_dff(x)
        # x_filp_conv, _ = self.conv1d_dinner(self.exit_proj_x(x_filp))
        
        
        y_1_fwd = self.ssm(x)
        y_1_bwd = self.ssm_exit(self.exit_proj_x(x_filp))
        # z branch
        z = F.silu(z) # [bs , seq_len , d_inner]  方向 ->
        z_filp = F.silu(z.flip([1])) # [bs , seq_len , d_ff]   方向 <-
        # z branch conv
        # z_conv, _ = self.conv1d_dff(z)
        # z_filp_conv, _ = self.conv1d_dinner(self.exit_proj_z(z_filp))
        # z_norm = F.silu(self.exit_proj_z(z)) # [bs , seq_len , d_inner]  方向 ->
        y_2_fwd = self.ssm(z)
        y_2_bwd = self.ssm_exit(self.exit_proj_z(z_filp))
        
        

        # print("1",y_2_fwd.shape==y_1_bwd.shape)
        # print("2",y_1_fwd.shape==y_2_bwd.shape)
        # 内部 -> <- 进行concat
        out_inner = self.out_inner_proj(torch.concat((y_1_fwd,y_2_fwd),dim=-1)) #+ hidden_states_fwd # [bs , seq_len , d_ff]
        # 外部 <- -> 进行concat
        out_exit = self.out_exit_proj(torch.concat((y_1_bwd,y_2_bwd),dim=-1)) #+ hidden_states_bwd # [bs , seq_len , d_inner]

        # d_inner 升维到 d_ff 通过一个activation后element-wise
        # output = out_inner * F.silu(self.up_dim(out_exit)) element-wise方式不合理
        # 残差的结构 把 hidden_states加入 且将out_exit进行翻转 看看是否在dstate为128时会低于3.80，在dstate为256时低于3.787
        output = hidden_states + torch.concat((out_inner,self.up_dim(out_exit.flip([1]))),dim=-1) # [bs , seq_len , 2 * d_ff] -> [bs , seq_len , d_ff]
        # output = self.Recurrent_proj(torch.concat((output,recurrent_states),dim=-1)) # [bs , seq_len , 2 * d_ff] -> [bs , seq_len , d_ff]
        # output = self.llm_input(output)

        # # concat prompt_mean (prompt_mean (bs * nvars, 1, embedding_dim)即这里的 bs, 1 , llm_dim)
        # llm_input = torch.cat([prompt_mean, output], dim=1) # [bs , seq_len+1 , llm_dim]
        # output = self.llm_model(inputs_embeds=llm_input).last_hidden_state
        # output = self.out_proj(output)

        '''
        output = torch.concat((out_inner,self.up_dim(out_exit)),dim=-1)
        output = self.llm_input(output) # [bs , seq_len , llm_model]
        output = self.llm_model(inputs_embeds=output).last_hidden_state
        output = self.out_proj(output)
        '''
        output = self.out_proj(output) # [bs , seq_len , d_model]

        return output
    
    def ssm(self, x):
        # x : [bs , seq_len, d_ff]

        # y : [bs , seq_len, d_ff]

        A = -torch.exp(self.A_log.float()) # [d_ff, d_state]

        deltaBCD = self.x_proj(x) # [bs * nvars, patch_num, dt_rank + 2 * d_state + d_ff]
        # [bs * nvars, patch_num, dt_rank], [bs * nvars, patch_num, d_state], [bs * nvars, patch_num, d_state], [bs * nvars, patch_num, d_ff]
        delta, B, C, D = torch.split(deltaBCD, [self.configs.dt_rank, self.configs.d_state, self.configs.d_state, self.configs.d_ff], dim=-1)
        delta = F.softplus(self.dt_proj(delta)) # [bs * nvars, patch_num, d_ff]

        if self.configs.pscan:
            y = self.selective_scan_inner(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y


    def ssm_exit(self, x):
        # x : [bs , seq_len, d_inner]

        # y : [bs , seq_len, d_inner]

        A = -torch.exp(self.A_log_exit.float()) # [d_inner, d_state]

        deltaBCD = self.x_proj_exit(x) # [bs * nvars, patch_num, dt_rank + 2 * d_state + d_inner]
        # [bs * nvars, patch_num, dt_rank], [bs * nvars, patch_num, d_state], [bs * nvars, patch_num, d_state], [bs * nvars, patch_num, d_ff]
        delta, B, C, D = torch.split(deltaBCD, [self.configs.dt_rank, self.configs.d_state, self.configs.d_state, self.d_inner], dim=-1)
        delta = F.softplus(self.dt_proj_exit(delta)) # [bs * nvars, patch_num, d_inner]

        if self.configs.pscan:
            y = self.selective_scan_External(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan_inner(self, x, delta, A, B, C, D):
        # x : [bs * nvars, patch_num, d_ff]
        # Δ : [bs * nvars, patch_num, d_ff]
        # A : [d_ff, d_state]
        # B : [bs * nvars, patch_num, d_state]
        # C : [bs * nvars, patch_num, d_state]
        # D : [bs * nvars, patch_num, d_ff]

        # y : [bs * nvars, patch_num, d_ff]


# bsnvar, patch_num, d_ff 56 31 512
# d_state 256
# deltaA.shape torch.Size([56, 31, 512, 256])
# deltaB.shape torch.Size([56, 31, 512, 256])

        bsnvar, patch_num, d_ff = x.shape
        d_state = C.shape[-1]
        # print("bsnvar, patch_num, d_ff",bsnvar, patch_num, d_ff)
        # print("d_state",d_state)
        # update
        deltaA = torch.exp(delta.unsqueeze(-1) * A) # [bs * nvars, patch_num, d_ff, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # [bs * nvars, patch_num, d_ff, d_state]
        # print("deltaA.shape",deltaA.shape)
        # print("deltaB.shape",deltaB.shape)
        BX = deltaB * (x.unsqueeze(-1)) # [bs * nvars, patch_num, d_ff, d_state]
        # print("BX.shape",BX.shape)
        hs = pscan(deltaA, BX) 
        # print("hs.shape",hs.shape) # hs.shape torch.Size([56, 31, 512, 256])
        # [bs * nvars, patch_num, d_ff, d_state] @ [bs * nvars, patch_num, d_state, 1] -> [bs * nvars, patch_num, d_ff]

        query = self.activate(self.queryinner_proj(x))
        key = self.activate(self.keyinner_proj(C))
        # print("query.shape",query.shape) # torch.Size([56, 31, 256])
        # print("key.shape",key.shape) # torch.Size([56, 31, 256])

        key_t = key.transpose(-2, -1)
        attn_scores = torch.bmm(query, key_t)  # [B, P, P]
        # print("attn_scores.shape",attn_scores.shape)
        # 除以缩放因子防止数值过大
        scale = d_state #** 0.5
        attn_scores = attn_scores / scale

        # softmax归一化，沿 d_state 维度（这里只有1维，等效对最后维度归一）
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, P, P]
        # print("attn_weights.shape",attn_weights.shape)
        
        # === 将attn_weights用于hs上的位置维度 === #
        # hs: [B, P, d_ff, d_state] → reshape 为 [B, d_ff, P, d_state]
        hs_reshape = hs.permute(0, 2, 1, 3).contiguous()          # [B, d_ff, P, d_state]

        # reshape attn_weights 为 [B, 1, P, P]，以便广播
        attn_weights = attn_weights.unsqueeze(1)                 # [B, 1, P, P]

        # 做位置维的加权求和：权重@value
        # (B, 1, P, P) @ (B, d_ff, P, d_state) → (B, d_ff, P, d_state)
        weighted_hs = torch.matmul(attn_weights, hs_reshape)     # [B, d_ff, P, d_state]

        # 最后转回来 [B, P, d_ff]
        y = weighted_hs.sum(dim=-1).permute(0, 2, 1).contiguous()  # [B, P, d_ff]
        # print("y.shape",y.shape) # y.shape torch.Size([56, 31, 512])
        y = y + D * x
        # print("y1.shape",y.shape) # y.shape torch.Size([56, 31, 512])
        return y

        y = (hs @ C.unsqueeze(-1)).squeeze(3)
        print("y.shape",y.shape) #y.shape torch.Size([56, 31, 512])
        # y = y + D * x
        print(weighted_hs.shape==y.shape)

        return weighted_hs
    
    def selective_scan_External(self, x, delta, A, B, C, D):
        # x : [bs * nvars, patch_num, d_ff]
        # Δ : [bs * nvars, patch_num, d_ff]
        # A : [d_ff, d_state]
        # B : [bs * nvars, patch_num, d_state]
        # C : [bs * nvars, patch_num, d_state]
        # D : [bs * nvars, patch_num, d_ff]

        # y : [bs * nvars, patch_num, d_ff]


# bsnvar, patch_num, d_ff 56 31 512
# d_state 256
# deltaA.shape torch.Size([56, 31, 512, 256])
# deltaB.shape torch.Size([56, 31, 512, 256])

        bsnvar, patch_num, d_ff = x.shape
        d_state = C.shape[-1]
        # print("bsnvar, patch_num, d_ff",bsnvar, patch_num, d_ff)
        # print("d_state",d_state)
        # update
        deltaA = torch.exp(delta.unsqueeze(-1) * A) # [bs * nvars, patch_num, d_ff, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # [bs * nvars, patch_num, d_ff, d_state]
        # print("deltaA.shape",deltaA.shape)
        # print("deltaB.shape",deltaB.shape)
        BX = deltaB * (x.unsqueeze(-1)) # [bs * nvars, patch_num, d_ff, d_state]
        # print("BX.shape",BX.shape)
        hs = pscan(deltaA, BX) 
        # print("hs.shape",hs.shape) # hs.shape torch.Size([56, 31, 512, 256])
        # [bs * nvars, patch_num, d_ff, d_state] @ [bs * nvars, patch_num, d_state, 1] -> [bs * nvars, patch_num, d_ff]

        query = self.activate(self.queryExternal_proj(x))
        key = self.activate(self.keyExternal_proj(C))
        # print("query.shape",query.shape) # torch.Size([56, 31, 256])
        # print("key.shape",key.shape) # torch.Size([56, 31, 256])

        key_t = key.transpose(-2, -1)
        attn_scores = torch.bmm(query, key_t)  # [B, P, P]
        # print("attn_scores.shape",attn_scores.shape)
        # 除以缩放因子防止数值过大
        scale = d_state #** 0.5
        attn_scores = attn_scores / scale

        # softmax归一化，沿 d_state 维度（这里只有1维，等效对最后维度归一）
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, P, P]
        # print("attn_weights.shape",attn_weights.shape)
        
        # === 将attn_weights用于hs上的位置维度 === #
        # hs: [B, P, d_ff, d_state] → reshape 为 [B, d_ff, P, d_state]
        hs_reshape = hs.permute(0, 2, 1, 3).contiguous()          # [B, d_ff, P, d_state]

        # reshape attn_weights 为 [B, 1, P, P]，以便广播
        attn_weights = attn_weights.unsqueeze(1)                 # [B, 1, P, P]

        # 做位置维的加权求和：权重@value
        # (B, 1, P, P) @ (B, d_ff, P, d_state) → (B, d_ff, P, d_state)
        weighted_hs = torch.matmul(attn_weights, hs_reshape)     # [B, d_ff, P, d_state]

        # 最后转回来 [B, P, d_ff]
        y = weighted_hs.sum(dim=-1).permute(0, 2, 1).contiguous()  # [B, P, d_ff]
        # print("y.shape",y.shape) # y.shape torch.Size([56, 31, 512])
        # y = y + D * x
        # print("y1.shape",y.shape) # y.shape torch.Size([56, 31, 512])
        return y


    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : [bs * nvars, patch_num, d_ff]
        # Δ : [bs * nvars, patch_num, d_ff]
        # A : [d_ff, d_state]
        # B : [bs * nvars, patch_num, d_state]
        # C : [bs * nvars, patch_num, d_state]
        # D : [bs * nvars, patch_num, d_ff]

        # y : [bs * nvars, patch_num, d_ff]

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # [bs * nvars, patch_num, d_ff, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # [bs * nvars, patch_num, d_ff, d_state]

        BX = deltaB * (x.unsqueeze(-1)) # [bs * nvars, patch_num, d_ff, d_state]

        h = torch.zeros(x.size(0), self.configs.d_ff, self.configs.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # [bs * nvars, patch_num, d_ff, d_state]
        # [bs * nvars, patch_num, d_ff, d_state] @ [bs * nvars, patch_num, d_state, 1] -> [bs * nvars, patch_num, d_ff, 1]
        y = (hs @ C.unsqueeze(-1)).squeeze(3)

        y = y + D * x

        return y



