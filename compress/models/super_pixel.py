import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F


class SuperPixelAttention_3(nn.Module):
    def __init__(self, dims, grid_size=[4, 4], n_iter=4, d_state=16, expand=2, **kwargs, ) -> None:
        super().__init__()
        # self.stoken_refine = MultiHeadSelfAttention(dims, heads=1)
        # self.stoken_refine = nn.MultiheadAttention()
        self.dwconv = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.grid_size = grid_size
        # print(self.grid_size)
        self.n_iter = n_iter
        self.scale = dims ** -0.5
        # self.ffn = FFN_2(dims)
        self.ln1 = nn.LayerNorm(dims)
        # self.linear = nn.Linear(dims,dims)

    def stoken_forward(self, x):
        # (b,h,w,c)
        # x = x.permute(0, 3, 1, 2).contiguous()
        x = x.contiguous()
        x0 = x + self.dwconv(x)
        x = self.ln1(x0.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        B, C, H0, W0 = x.shape
        # print("x_ori.shape",x.shape)

        h, w = self.grid_size

        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        _, _, H, W = x.shape
        p, q = H // h, W // w

        # rearrange the tokens
        tokens = rearrange(x, "b c (y h) (x w) -> b (y x) (h w) c", y=p, x=q)

        # compute the initial super tokens
        stokens = F.adaptive_avg_pool2d(x, (p, q))

        # compute the associations iteratively
        with torch.no_grad():
            for idx in range(self.n_iter):
                # extract the 9 surrounding super tokens
                stokens = F.unfold(stokens, kernel_size=3, padding=1)
                stokens = stokens.transpose(1, 2).reshape(B, p * q, C, 9)

                # compute sparse associations (B, p*q, h*w, 9) b pq hw c
                # tokens b pq hw c
                # stokens b pq c 9
                # association b pq hw 9
                association = tokens @ stokens * self.scale
                association = association.softmax(-1)

                # prepare for association normalization
                association_sum = association.sum(2).transpose(1, 2).reshape(B, 9, p * q)
                association_sum = F.fold(association_sum, output_size=(p, q), kernel_size=3, padding=1, stride=1)

                if idx < self.n_iter - 1:
                    # compute super tokens
                    stokens = tokens.transpose(-1, -2) @ association
                    stokens = stokens.permute(0, 2, 3, 1).reshape(B, C * 9, p * q)
                    stokens = F.fold(stokens, output_size=(p, q), kernel_size=3, padding=1, stride=1)
                    stokens = stokens / (association_sum + 1e-12)

        # MHSA for super tokens
        stokens = tokens.transpose(-1, -2) @ association
        stokens = stokens.permute(0, 2, 3, 1).reshape(B, C * 9, p * q)
        stokens = F.fold(stokens, output_size=(p, q), kernel_size=3, padding=1, stride=1)
        stokens = stokens / (association_sum.detach() + 1e-12)
        # print("stokens1.shape",stokens.shape)
        # stokens = self.stoken_refine(stokens.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        # print("stokens2.shape",stokens.shape)



        # map super tokens back to tokens
        # stokens = F.unfold(stokens, kernel_size=3, padding=1)
        # stokens = stokens.transpose(1, 2).reshape(B, p * q, C, 9)
        # tokens = stokens @ association.transpose(-1, -2)

        # Rearrange tokens to get the final output
        # output = rearrange(tokens, "b (y x) c (h w) -> b c (y h) (x w)", y=p, x=q, h=h, w=w)
        # if pad_r > 0 or pad_b > 0:
        #     output = output[:, :, :H0, :W0]
        # print("output.shape",output.shape)
        # output = output + x0
        # output = self.ffn(output)+output
        # return stokens
        return stokens
