import numpy as np
import torch
import ruptures as rpt
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller

class TimeSeriesStatsExtractor:
    def __init__(self, seq_len, pred_len, description="", top_k=5):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.description = description
        self.top_k = top_k

    def calcute_lags(self, x_enc):
        """
        x_enc: shape (B*N, T, 1)
        Return: top-k lags list for each sequence (shape: B*N, k)
        """
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)  # [B*N, T]
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags.tolist()  # shape: [B*N, top_k]

    def _extract_distribution_features(self, x):
        std_dev = np.std(x)
        skew_val = skew(x)
        kurt_val = kurtosis(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        return std_dev, skew_val, kurt_val, iqr

    def _extract_stationarity_features(self, x, window=10):
        if np.all(x == x[0]):
            # 常数序列：返回默认值
            # print("常数")
            adf_p = 1.0  # 完全非平稳
            mean_std = 0.0
            var_std = 0.0
            slope = 0.0
        else:
            # ADF 单位根检验
            adf_p = adfuller(x)[1]

            # 滑动窗口均值方差稳定性
            means, vars_ = [], []
            for i in range(0, len(x) - window + 1, window):
                seg = x[i:i+window]
                means.append(np.mean(seg))
                vars_.append(np.var(seg))
            mean_std = np.std(means)
            var_std = np.std(vars_)

            # 总体趋势斜率
            slope = np.polyfit(np.arange(len(x)), x, 1)[0]

        return adf_p, mean_std, var_std, slope

    def _detect_structural_breaks(self, x, penalty=5):
        try:
            algo = rpt.Pelt(model="l2").fit(x)
            break_locs = algo.predict(pen=penalty)[:-1]
        except:
            break_locs = []
        return len(break_locs), break_locs

    def extract_prompts(self, x_enc):
        """
        x_enc: Tensor (B, T, N)
        Returns: List of prompt strings
        """
        B, T, N = x_enc.shape
        x_reshaped = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        x_np = x_reshaped.squeeze(-1).detach().cpu().numpy()

        # 统计量
        min_values = torch.min(x_reshaped, dim=1)[0]
        max_values = torch.max(x_reshaped, dim=1)[0]
        medians = torch.median(x_reshaped, dim=1).values
        trends = x_reshaped.diff(dim=1).sum(dim=1)

        lags = self.calcute_lags(x_reshaped)

        prompts = []
        for b in range(B * N):
            x_b = x_np[b]

            # 基本统计量
            min_val = float(min_values[b].item())
            max_val = float(max_values[b].item())
            median_val = float(medians[b].item())
            trend_val = float(trends[b].item())
            lags_str = str(lags[b])

            # 分布特征
            std_dev, skew_val, kurt_val, iqr = self._extract_distribution_features(x_b)

            # 平稳性特征
            adf_p, mean_std, var_std, slope = self._extract_stationarity_features(x_b)

            # 结构突变
            n_bkps, break_locs = self._detect_structural_breaks(x_b)
            if n_bkps > 0:
                breaks_str = f"{n_bkps} structural breakpoints detected at time steps {break_locs}"
            else:
                breaks_str = "no clear structural breakpoints detected"

            # prompt = (
            #     f"<|start_prompt|>Dataset description: {self.description} "
            #     f"Task description: Task objective is to forecast the next {self.pred_len} time steps using historical {self.seq_len} time steps information; "
            #     "Input statistics: "
            #     f"min value {min_val:.4f}, "
            #     f"max value {max_val:.4f}, "
            #     f"median value {median_val:.4f}, "
            #     f"trend sum {trend_val:.4f}, "
            #     f"standard deviation {std_dev:.4f}, "
            #     f"skewness {skew_val:.4f}, "
            #     f"kurtosis {kurt_val:.4f}, "
            #     f"interquartile range {iqr:.4f}, "
            #     f"stationarity properties: ADF p-value {adf_p:.4f}, mean stability std {mean_std:.4f}, variance stability std {var_std:.4f}, trend slope {slope:.4f}; "
            #     f"structural break analysis: {breaks_str}; "
            #     f"the inputs are processed twice, one without flipping and the other with flipping (flipping in the time dimension), "
            #     f"top {self.top_k} lags are : {lags_str}<|<end_prompt>|>"
            # )


            # basic + dist
            prompt = (
                f"<|start_prompt|>Dataset description: {self.description} "
                f"Task description: Task objective is to forecast the next {self.pred_len} time steps using historical {self.seq_len} time steps information; "
                "Input statistics: "
                f"standard deviation {std_dev:.4f}, "
                f"skewness {skew_val:.4f}, "
                f"kurtosis {kurt_val:.4f}, "
                f"interquartile range {iqr:.4f}, "
                f"stationarity properties: ADF p-value {adf_p:.4f}, mean stability std {mean_std:.4f}, variance stability std {var_std:.4f}, trend slope {slope:.4f}; "
                f"structural break analysis: {breaks_str}; "
                f"the inputs are processed twice, one without flipping and the other with flipping (flipping in the time dimension), "
                f"top {self.top_k} lags are : {lags_str}<|<end_prompt>|>"
            )

            # with 结构突变
            # prompt = (
            #     f"<|start_prompt|>Dataset description: {self.description} "
            #     f"Task description: Task objective is to forecast the next {self.pred_len} time steps using historical {self.seq_len} time steps information; "
            #     "Input statistics: "
            #     f"standard deviation {std_dev:.4f}, "
            #     f"skewness {skew_val:.4f}, "
            #     f"kurtosis {kurt_val:.4f}, "
            #     f"interquartile range {iqr:.4f}, "
            #     f"the inputs are processed twice, one without flipping and the other with flipping (flipping in the time dimension), "
            #     f"top {self.top_k} lags are : {lags_str}<|<end_prompt>|>"
            # )

            # with 平稳性
            # prompt = (
            #     f"<|start_prompt|>Dataset description: {self.description} "
            #     f"Task description: Task objective is to forecast the next {self.pred_len} time steps using historical {self.seq_len} time steps information; "
            #     "Input statistics: "
            #     f"stationarity properties: ADF p-value {adf_p:.4f}, mean stability std {mean_std:.4f}, variance stability std {var_std:.4f}, trend slope {slope:.4f}; "
            #     f"structural break analysis: {breaks_str}; "
            #     f"the inputs are processed twice, one without flipping and the other with flipping (flipping in the time dimension), "
            #     f"top {self.top_k} lags are : {lags_str}<|<end_prompt>|>"
            # )

            # with 分布 
            # prompt = (
            #     f"<|start_prompt|>Dataset description: {self.description} "
            #     f"Task description: Task objective is to forecast the next {self.pred_len} time steps using historical {self.seq_len} time steps information; "
            #     "Input statistics: "
            #     f"standard deviation {std_dev:.4f}, "
            #     f"skewness {skew_val:.4f}, "
            #     f"kurtosis {kurt_val:.4f}, "
            #     f"interquartile range {iqr:.4f}, "
            #     f"the inputs are processed twice, one without flipping and the other with flipping (flipping in the time dimension), "
            #     f"top {self.top_k} lags are : {lags_str}<|<end_prompt>|>"
            # )


            # with 分布 平稳性  结构突变
            # prompt = (
            #     f"<|start_prompt|>Dataset description: {self.description} "
            #     f"Task description: Task objective is to forecast the next {self.pred_len} time steps using historical {self.seq_len} time steps information; "
            #     "Input statistics: "
            #     f"standard deviation {std_dev:.4f}, "
            #     f"skewness {skew_val:.4f}, "
            #     f"kurtosis {kurt_val:.4f}, "
            #     f"interquartile range {iqr:.4f}, "
            #     f"stationarity properties: ADF p-value {adf_p:.4f}, mean stability std {mean_std:.4f}, variance stability std {var_std:.4f}, trend slope {slope:.4f}; "
            #     f"structural break analysis: {breaks_str}; "
            #     f"the inputs are processed twice, one without flipping and the other with flipping (flipping in the time dimension), "
            #     f"top {self.top_k} lags are : {lags_str}<|<end_prompt>|>"
            # )

            # with basic
            # prompt = (
            #     f"<|start_prompt|>Dataset description: {self.description} "
            #     f"Task description: Task objective is to forecast the next {self.pred_len} time steps using historical {self.seq_len} time steps information; "
            #     "Input statistics: "
            #     f"min value {min_val:.4f}, "
            #     f"max value {max_val:.4f}, "
            #     f"median value {median_val:.4f}, "
            #     f"trend sum {trend_val:.4f}, "
            #     f"the inputs are processed twice, one without flipping and the other with flipping (flipping in the time dimension), "
            #     f"<|<end_prompt>|>"
            # )

            prompts.append(prompt)

        return prompts
