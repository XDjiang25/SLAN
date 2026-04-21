import math
from typing import Optional
from torch import einsum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layers.RevIN import RevIN
from layers.Embed import PatchEmbedding, MultiScalePatchEmbedding
from einops import rearrange, repeat
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoConfig, AutoModel, AutoTokenizer
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from layers.SLANencoder import InMambaEncoder
from layers.prompt_construct import TimeSeriesStatsExtractor
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    



class Model(nn.Module):

#just make sure d_model * expand / headdim = multiple of 8
# headdim = 64
# 128 
    def __init__(self, configs):
        super(Model, self).__init__()

        self.revin = configs.revin
        if configs.revin==1:
            self.revin_layer = RevIN(configs.enc_in)
        self.e_layers = configs.e_layers
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # patch
        self.patch_len = configs.patch_size   
        self.stride = configs.patch_stride
        # self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        # multisclae
        # self.patch_lens = [32, 4]
        # self.patch_lens = [64, 32]
        self.patch_lens = [32, 4] # ILI 36_24 
        self.patch_nums = sum([int((configs.seq_len - patch_len) / self.stride + 2) for patch_len in self.patch_lens])

        self.multiscale_patchembedding = MultiScalePatchEmbedding(configs.d_model, patch_lens=self.patch_lens, stride=self.stride, dropout=configs.dropout)
        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout)
        #patch_num = (L - patch_size) // stride + 1
        self.head_nf = configs.d_model * self.patch_nums
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                        head_dropout=configs.head_dropout)

        self.class_strategy = configs.class_strategy

        self.encoder = InMambaEncoder(configs)
        

        self.lin1 = nn.Linear(configs.enc_in, configs.d_model)  #将seq_len -> n1  B L M -> B M n1
        self.output_proj = nn.Linear(configs.d_model, configs.enc_in, bias=False)
        self.llm_input = nn.Linear(configs.llm_dim, configs.d_model,  bias=False)
        # self.llm_output = nn.Linear(configs.llm_dim, configs.d_model, bias=False)

        # MetaEnhanced
        self.qwen2_config = AutoConfig.from_pretrained('/remote-home/share/dmb_nas/jxd/language_model/openai-community/gpt2')
        self.qwen2_config.num_hidden_layers = configs.llm_layers  # 设置隐藏层数
        # self.qwen2_config.output_attentions = True
        self.qwen2_config.output_hidden_states = True
        self.Meta_encoder = AutoModel.from_pretrained(
            '/remote-home/share/dmb_nas/jxd/language_model/openai-community/gpt2',
            trust_remote_code=True,
            local_files_only=True,
            config=self.qwen2_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            '/remote-home/share/dmb_nas/jxd/language_model/openai-community/gpt2',
            trust_remote_code=True,
            local_files_only=True
        )

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        self.word_embeddings = self.Meta_encoder.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        # self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment. The time resolution is hourly'
        # self.description = 'Weather is recorded every 10 minutes for the 2020 whole year, which contains 21 meteorological indicators, such as air temperature, humidity, etc.'
        # self.description = 'Daily exchange rates were collected for eight countries from 1990 to 2016.'
        # self.description = 'Weather is recorded every 10 minutes for the 2020 whole year, which contains 21 meteorological indicators, such as air temperature, humidity, etc.'
        self.description = configs.description
        self.top_k = 5

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.revin==1:
            x_enc = self.revin_layer(x_enc,'norm')
        else:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, T, N = x_enc.shape # B L N
   
        # B: batch_size;    E: d_model; 
        # T: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        extractor = TimeSeriesStatsExtractor(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            description=self.description,
            top_k=5
        )
        prompt = extractor.extract_prompts(x_enc)
        # print("prompt",prompt)
        # x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # min_values = torch.min(x_enc, dim=1)[0]
        # max_values = torch.max(x_enc, dim=1)[0]
        # medians = torch.median(x_enc, dim=1).values
        # lags = self.calcute_lags(x_enc)
        # trends = x_enc.diff(dim=1).sum(dim=1)

        # prompt = []
        # for b in range(x_enc.shape[0]):
        #     min_values_str = str(min_values[b].tolist()[0])
        #     max_values_str = str(max_values[b].tolist()[0])
        #     median_values_str = str(medians[b].tolist()[0])
        #     lags_values_str = str(lags[b].tolist())
        #     prompt_ = (
        #         f"<|start_prompt|>Dataset description: {self.description}"
        #         f"Task description: Task objective is to forecast the next {str(self.pred_len)} time steps using historical {str(self.seq_len)} time steps information; "
        #         "Input statistics: "
        #         f"min value {min_values_str}, "
        #         f"max value {max_values_str}, "
        #         f"median value {median_values_str}, "
        #         f"the inputs are processed twice, one without flipping and the other with flipping (flipping in the time dimension), "
        #         f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
        #     )

        #     prompt.append(prompt_)
        # #prompt = [str1, str2, ..., strN]  # 长度为 B*N（i.e. x_enc.shape[0]）   
        # x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()


        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        # 形状变为 (B*N, prompt_token_len)
        prompt_embeddings = self.Meta_encoder.get_input_embeddings()(prompt.to(x_enc.device))  # (bs * nvars, prompt_token, embedding_dim)
        prompt_embeddings = self.llm_input(prompt_embeddings)
        # AvgPooling
        prompt_mean = prompt_embeddings.mean(dim=1)  # (bs * nvars, embedding_dim)
        prompt_mean = prompt_mean.unsqueeze(1) # → (bs * nvars, 1, embedding_dim)
        
        # Patch Embedding
        # x_enc = x_enc.permute(0, 2, 1) # B N L
        # print("输入",x_enc.shape) # 输入 torch.Size([16, 7, 36])
        enc_out, n_vars = self.multiscale_patchembedding(x_enc.to(torch.bfloat16))
        patch_num = enc_out.shape[1]
        # print("multiscale-n_vars",n_vars)
        # print("multiscale-enc_out.shape",enc_out.shape) # torch.Size([576, 6, 256])
        # print("patch_num",patch_num)
        # print("self.patch_nums",self.patch_nums)
        # enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        # [bs * nvars, patch_num, d_model]
        # print("multiscale-n_vars",n_vars)
        # print("multiscale-enc_out.shape",enc_out.shape) # torch.Size([576, 6, 256])
        
        mamba_input = torch.cat([prompt_mean, enc_out], dim=1) # [bs , seq_len+1 , d_model]
        mamba_input = torch.cat([mamba_input, prompt_mean], dim=1) # [bs , seq_len+1+1 , d_model]
        enc_out = self.encoder(mamba_input) # [bs * nvars, patch_num + 1, d_model] 因为concat了prompt
        # enc_out = self.hybrid(enc_out)

        # enc_out = self.encoder(enc_out) # [bs * nvars, patch_num + 1, d_model] 因为concat了prompt
        # enc_out = self.Meta_encoder(inputs_embeds=llm_input).last_hidden_state
        # enc_out = self.llm_output(enc_out) # [bs , seq_len+1 , d_model]
        # print("mamba enc_out.shape",enc_out.shape)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # [bs, nvars, d_model, patch_num]

        # print("reshape enc_out.shape",enc_out.shape) #reshape enc_out.shape torch.Size([16, 36, 6, 256])
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out[:, :, :, -patch_num:])  # dec_out: [bs, nvars, target_window]
        # print("dec_out.shape",dec_out.shape)
        # dec_out = self.output_proj(Inner_output)
        dec_out = dec_out.permute(0, 2, 1)
        # print(dec_out.shape)

        if self.revin ==1:
            dec_out = self.revin_layer(dec_out,'denorm')

        else:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        return dec_out[:, -self.pred_len:, :]  # [B, L, D] 
    
    # def calcute_lags(self, x_enc):
    #     q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
    #     k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
    #     res = q_fft * torch.conj(k_fft)
    #     corr = torch.fft.irfft(res, dim=-1)
    #     mean_value = torch.mean(corr, dim=1)
    #     _, lags = torch.topk(mean_value, self.top_k, dim=-1)
    #     return lags