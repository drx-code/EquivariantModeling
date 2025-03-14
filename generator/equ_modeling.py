from functools import partial
from typing import Optional

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block, Attention
from timm.layers import Mlp
from generator.diffloss import DiffLoss
from timm.layers import DropPath

class MaskedAttention(Attention):
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask[None,None,:,:]
            )
            assert mask.dtype == torch.bool
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill(~mask[None, None, :, :], float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MaskedBlock(Block):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            proj_drop,
            attn_drop,
            init_values,
            drop_path,
            act_layer,
            norm_layer,
            mlp_layer,
        )
        # Replace the Attention instance with MaskedAttention
        self.attn = MaskedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), mask=mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class MaskedAttentionRoPE(Attention):
    def __init__(self, cls_token_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_token_num = cls_token_num
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, freq_cis=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # B, num_heads, N, head_dim
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        q = q.permute(0, 2, 1, 3) 
        k = k.permute(0, 2, 1, 3)
        q = apply_rotary_emb(q, freq_cis, cls_token_num=self.cls_token_num).permute(0, 2, 1, 3) # B, 16
        k = apply_rotary_emb(k, freq_cis, cls_token_num=self.cls_token_num).permute(0, 2, 1, 3)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask[None,None,:,:]
            )
            assert mask.dtype == torch.bool
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill(~mask[None, None, :, :], float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MaskedBlockRoPE(Block):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            cls_token_num: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            proj_drop,
            attn_drop,
            init_values,
            drop_path,
            act_layer,
            norm_layer,
            mlp_layer,
        )
        # Replace the Attention instance with MaskedAttention
        self.attn = MaskedAttentionRoPE(
            cls_token_num,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, freq_cis: torch.Tensor = None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), mask=mask, freq_cis=freq_cis)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class MaskedAttentionCrossRoPE(Attention):
    def __init__(self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm):
        super().__init__(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer)
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor, freq_cis=None) -> torch.Tensor:
        B, N, C = x.shape
        B, cond_len, C = cond.shape
        q = self.qkv(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B, num_heads, N, head_dim 
        k = self.k(cond).reshape(B, cond_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B, num_heads, N, head_dim 
        v = self.v(cond).reshape(B, cond_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # B, num_heads, N, head_dim 
        q, k = self.q_norm(q), self.k_norm(k)
        
        q = q.permute(0, 2, 1, 3)  # B,  N, num_heads, head_dim 
        q = apply_rotary_emb(q, freq_cis, cls_token_num=0).permute(0, 2, 1, 3) # B, num_heads, N, head_dim 

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=None
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x    
    
class MaskedBlockRoPECrossAttn(Block):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            cls_token_num: int,
            token_num: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            proj_drop,
            attn_drop,
            init_values,
            drop_path,
            act_layer,
            norm_layer,
            mlp_layer,
        )
        # Replace the Attention instance with MaskedAttention
        self.attn = MaskedAttentionRoPE(
            0,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.attn2 = MaskedAttentionCrossRoPE(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.cross_norm = nn.LayerNorm(normalized_shape=dim)
        self.cross_cond_norm = nn.LayerNorm(normalized_shape=dim)
        self.cls_token_num = cls_token_num
        self.cross_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cross_ls = nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, freq_cis: torch.Tensor = None) -> torch.Tensor:
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), mask=mask, freq_cis=freq_cis)))
        cond_token, img_token = x[:, :self.cls_token_num], x[:, self.cls_token_num:]
        
        # self attn
        img_token_norm1 = self.norm1(img_token)
        # cond_token, img_token = x_norm1[:, :self.cls_token_num], x_norm1[:, self.cls_token_num:]
        # a = self.attn(img_token, mask=mask, freq_cis=freq_cis)
        img_token_attn1 = self.attn(img_token_norm1, mask=mask, freq_cis=freq_cis)
        img_token = img_token + self.drop_path1(self.ls1(img_token_attn1))
        # x = x + self.drop_path1(self.ls1(torch.concat([cond_token, a], dim=1)))
        
        # cross attn
        cross_normed_img_token, cross_normed_cond_token = self.cross_norm(img_token), self.cross_cond_norm(cond_token)
        # cond_token, img_token = cross_normed[:, :self.cls_token_num], cross_normed[:, self.cls_token_num:]
        img_token_attn2 = self.attn2.forward(x=cross_normed_img_token, cond=cross_normed_cond_token, freq_cis=freq_cis)
        img_token = img_token + self.cross_drop_path(self.cross_ls(img_token_attn2))
        # x = x + self.cross_drop_path(self.cross_ls(torch.concat([cond_token, a], dim=1)))
        
        x = torch.concat([cond_token, img_token], dim=1)
        
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class EquivariantGenerativeModel(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, token_num=128, 
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 cond_length=0,
                 shift_num=16,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        self.vae_embed_dim = vae_embed_dim
        self.num_heads = encoder_num_heads
        self.img_size = img_size
        self.seq_len = token_num
        self.token_embed_dim = vae_embed_dim
        self.grad_checkpointing = grad_checkpointing
        self.cond_length = cond_length
        self.shift_num =  self.seq_len if shift_num is None else shift_num

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # encoder
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            MaskedBlockRoPECrossAttn(encoder_embed_dim, encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, cls_token_num=buffer_size, token_num=token_num,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth+decoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing,
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, class_embedding):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1) + self.encoder_pos_embed_learned

        # encoder position embedding
        x = self.z_proj_ln(x)

        # dropping
        x = x.reshape(bsz, -1, embed_dim)
        
        mask = torch.zeros(seq_len, seq_len).to(x.device)
        mask[:, :] = 1 - (torch.triu(torch.ones(seq_len, seq_len), diagonal=1)).to(x.device)
        if self.cond_length > 0:
            mask[:, :] -= 1 - (torch.triu(torch.ones(seq_len, seq_len), diagonal=-1*self.cond_length)).to(x.device)
        mask = mask.bool()
        
        freq_cis = precompute_freqs_cis(x.shape[1] - self.buffer_size, x.shape[-1] // self.num_heads, cls_token_num=0, training=self.training, shift_num=self.shift_num)
        freq_cis = freq_cis.to(x.device)
        
        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x, mask, freq_cis)
        else:
            for block in self.encoder_blocks:
                x = block(x, mask, freq_cis)
        x = self.encoder_norm(x)
        x = x[:, self.buffer_size-1:-1]
        return x

    def forward_loss(self, z, target):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        loss = self.diffloss.forward(z=z, x0=target)
        return loss
    
    def forward(self, imgs, labels):
        # class embed
        class_embedding = self.class_emb(labels)
        x = imgs
        gt_latents = x.clone().detach()

        # mae encoder
        x = self.forward_encoder(x, class_embedding)

        # diffloss
        loss = self.forward_loss(z=x, target=gt_latents)

        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False, gt_latents=None):
        # init and sample generation orders
        if gt_latents is None:
            tokens = torch.zeros(bsz, num_iter, self.token_embed_dim).cuda()

            indices = list(range(num_iter))
            if progress:
                indices = tqdm(indices)
            # generate latents
            for step in indices:
                
                cur_tokens = tokens.clone()

                # class embedding and CFG
                if labels is not None:
                    class_embedding = self.class_emb(labels)
                else:
                    class_embedding = self.fake_latent.repeat(bsz, 1)
                if not cfg == 1.0:
                    tokens = torch.cat([tokens, tokens], dim=0)
                    class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)

                # mae encoder
                x = self.forward_encoder(tokens, class_embedding)

                # sample token latents for this step
                z = x[:, step]
                # cfg schedule follow Muse
                if cfg_schedule == "linear":
                    cfg_iter = 1 + (cfg - 1) * (step + 1.) / num_iter
                elif cfg_schedule == "constant":
                    cfg_iter = cfg
                else:
                    raise NotImplementedError
                
                sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
                if not cfg == 1.0:
                    sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                
                cur_tokens[:, step] = sampled_token_latent
                tokens = cur_tokens.clone()
        else:
            tokens = torch.zeros(bsz, num_iter, self.token_embed_dim).cuda()
            tokens[:, :gt_latents.shape[1]] = gt_latents[:, :]

            indices = list(range(gt_latents.shape[1], num_iter))
            if progress:
                indices = tqdm(indices)
            # generate latents
            for step in indices:
                
                cur_tokens = tokens.clone()

                # class embedding and CFG
                if labels is not None:
                    class_embedding = self.class_emb(labels)
                else:
                    class_embedding = self.fake_latent.repeat(bsz, 1)
                if not cfg == 1.0:
                    tokens = torch.cat([tokens, tokens], dim=0)
                    class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)

                # mae encoder
                x = self.forward_encoder(tokens, class_embedding)

                # sample token latents for this step
                z = x[:, step]
                # cfg schedule follow Muse
                if cfg_schedule == "linear":
                    cfg_iter = 1 + (cfg - 1) * (self.seq_len - (num_iter - step - 1)) / self.seq_len
                elif cfg_schedule == "constant":
                    cfg_iter = cfg
                else:
                    raise NotImplementedError
                
                sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
                if not cfg == 1.0:
                    sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                cur_tokens[:, step] = sampled_token_latent
                tokens = cur_tokens.clone()

        tokens = tokens.permute(0, 2, 1)[:,:,None,:]
        return tokens
    
#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120, training=True, shift_num=None):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    bias = torch.randint(0, seq_len if shift_num is None else shift_num, (1,), device=freqs.device) if training \
        else torch.zeros([1], device=freqs.device)
    t = torch.arange(seq_len, device=freqs.device) + bias

    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, cls_token_num=0):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[:, cls_token_num:, :, :, 0] * freqs_cis[:, cls_token_num:, :, :, 0] - xshaped[:, cls_token_num:, :, :, 1] * freqs_cis[:, cls_token_num:, :, :, 1],
            xshaped[:, cls_token_num:, :, :, 1] * freqs_cis[:, cls_token_num:, :, :, 0] + xshaped[:, cls_token_num:, :, :, 0] * freqs_cis[:, cls_token_num:, :, :, 1],
    ], dim=-1)
    x_out2 = torch.concat([xshaped[:, :cls_token_num], x_out2], dim=1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

def tiny_model(**kwargs):
    model = EquivariantGenerativeModel(
        encoder_embed_dim=256, encoder_depth=4, encoder_num_heads=4,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def small_model(**kwargs):
    model = EquivariantGenerativeModel(
        encoder_embed_dim=512, encoder_depth=8, encoder_num_heads=8,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def base_model(**kwargs):
    model = EquivariantGenerativeModel(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def large_model(**kwargs):
    model = EquivariantGenerativeModel(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def huge_model(**kwargs):
    model = EquivariantGenerativeModel(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
