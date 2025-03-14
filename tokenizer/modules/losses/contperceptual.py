import torch
import torch.nn as nn

from tokenizer.modules.losses.vqperceptual import * 
from tokenizer.modules.encoder_utils import load_encoders
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from tokenizer.modules.discriminator.dino_disc import DinoDisc as DINODiscriminator
from tokenizer.modules.discriminator.diffaug import DiffAug

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    x = (x + 1.) /  2.
    if 'clip' in enc_type:
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    return x

class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0, semantic_iter_start=0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False, disc_ndf=64, semantic_loss_w=1., disc_type='patchgan',
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.disc_type = disc_type
        if disc_type == 'patchgan':
            self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                    n_layers=disc_num_layers,
                                    use_actnorm=use_actnorm,
                                    ndf=disc_ndf,
                                ).apply(weights_init)
        elif disc_type == "dino":
            self.discriminator = DINODiscriminator(norm_type='bn')
            self.daug = DiffAug(prob=1, cutout=0.2)
        else:
            raise NotImplementedError
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.l2loss = nn.MSELoss()
        
        encoders,_,_ = load_encoders(enc_type = "dinov2-vit-b", device="cuda")
        self.encoder = encoders[0]
        self.semantic_loss_w = semantic_loss_w
        self.semantic_iter_start = semantic_iter_start

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, semantic_output, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        
        # semantic loss
        dino_input = preprocess_raw_image(inputs, enc_type="dinov2-vit-b")
        
        dino_features = self.encoder.forward_features(dino_input)['x_norm_patchtokens']
        dino_features = dino_features.transpose(1, 2)
        semantic_output = semantic_output.reshape(semantic_output.shape[0], semantic_output.shape[1], -1)
        
        semantic_loss = torch.pow(semantic_output.contiguous() - dino_features.contiguous(), 2)
        semantic_loss = torch.sum(semantic_loss) / semantic_loss.shape[0]
        semantic_factor = adopt_weight(self.semantic_loss_w, global_step, threshold=self.semantic_iter_start)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                if self.disc_type == "patchgan":
                    logits_fake = self.discriminator(reconstructions.contiguous())
                elif self.disc_type == "dino":
                    logits_fake = self.discriminator(self.daug.aug(reconstructions.contiguous(), warmup_blur_schedule=0))
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss + semantic_factor * semantic_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   "{}/semantic_loss".format(split): semantic_loss.detach().mean(),
                   "{}/semantic_factor".format(split): torch.tensor(semantic_factor),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                if self.disc_type == "patchgan":
                    logits_real = self.discriminator(inputs.contiguous().detach())
                    logits_fake = self.discriminator(reconstructions.contiguous().detach())
                elif self.disc_type == "dino":
                    logits_real = self.discriminator(self.daug.aug(inputs.contiguous().detach(), warmup_blur_schedule=0))
                    logits_fake = self.discriminator(self.daug.aug(reconstructions.contiguous().detach(), warmup_blur_schedule=0))
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
