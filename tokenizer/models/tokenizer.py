import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import importlib
from packaging import version

from tokenizer.modules.diffusionmodules.model import EquivariantTokenizerEncoder1D, EquivariantTokenizerDecoder1D, EquivariantTokenizerSemanticDecoder1D
from tokenizer.modules.distributions.distributions import GaussianDistribution
from torch.optim.lr_scheduler import LambdaLR

from contextlib import contextmanager
from tokenizer.ema import LitEma

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

# Add ema
class EquivariantTokenizer(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 sematicconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 use_ema=True,
                 ema_decay=0.99,
                 post_channels=4096,
                 scheduler_config=None,
                 stage2=False,
                 std=2.5,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = EquivariantTokenizerEncoder1D(**ddconfig)
        self.decoder = EquivariantTokenizerDecoder1D(**ddconfig)
        self.semantic_decoder = EquivariantTokenizerSemanticDecoder1D(**sematicconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quant_layer = torch.nn.Linear(post_channels, embed_dim) 
        self.post_quant_layer = torch.nn.Linear(embed_dim, post_channels)
        if True:
            self.quant_layer = torch.nn.Conv1d(post_channels * 2 if ddconfig["double_enc"] else post_channels, embed_dim, kernel_size=3, padding=1, stride=1, padding_mode="reflect") 
            self.post_quant_layer = torch.nn.Conv1d(embed_dim, post_channels, kernel_size=3, padding=1, stride=1, padding_mode="reflect")
        self.embed_dim = embed_dim
        self.stage2 = stage2
        self.std = std
        if stage2:
            print("Only finetune Decoder")
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_encoder = LitEma(self.encoder, ema_decay)
            self.ema_decoder = LitEma(self.decoder, ema_decay)
            self.ema_semantic_decoder = LitEma(self.semantic_decoder, ema_decay)
            self.ema_quant_layer = LitEma(self.quant_layer, ema_decay) 
            self.ema_post_quant_layer = LitEma(self.post_quant_layer, ema_decay) 
            
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
            
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        if self.stage2:
            self.ema_encoder.copy_to(self.encoder)
            self.ema_decoder.copy_to(self.decoder)
            self.ema_post_quant_layer.copy_to(self.post_quant_layer) 
            self.ema_quant_layer.copy_to(self.quant_layer)
            self.ema_semantic_decoder.copy_to(self.semantic_decoder)   
        print(f"Restored from {path}")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.ema_encoder.store(self.encoder.parameters())
            self.ema_encoder.copy_to(self.encoder)
            self.ema_decoder.store(self.decoder.parameters())
            self.ema_decoder.copy_to(self.decoder)
            self.ema_post_quant_layer.store(self.post_quant_layer.parameters())
            self.ema_post_quant_layer.copy_to(self.post_quant_layer)   
            self.ema_quant_layer.store(self.quant_layer.parameters())
            self.ema_quant_layer.copy_to(self.quant_layer)  
            self.ema_semantic_decoder.store(self.semantic_decoder.parameters()) 
            self.ema_semantic_decoder.copy_to(self.semantic_decoder)                       
    
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                # self.model_ema.restore(self.model.parameters())
                self.ema_encoder.restore(self.encoder.parameters())
                self.ema_decoder.restore(self.decoder.parameters())
                self.ema_post_quant_layer.restore(self.post_quant_layer.parameters())
                self.ema_quant_layer.restore(self.quant_layer.parameters())
                self.ema_semantic_decoder.restore(self.semantic_decoder.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")


    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_layer(h.permute(0,2,1))
        moments = moments.permute(0,2,1)
        posterior = GaussianDistribution(moments, std=self.std)
        return posterior

    def decode(self, z):
        z = self.post_quant_layer(z.permute(0,2,1))
        dec, semantic_map  = self.decoder(z.permute(0,2,1))
        semantic_dec = self.semantic_decoder(semantic_map)
        return dec, semantic_dec

    def forward(self, input, sample_posterior=True):
        if self.use_scheduler and self.training:
            lr = self.optimizers()[0].param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec, semantic_dec = self.decode(z)
        return dec, posterior, semantic_dec
    
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        # x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior, semantic_dec = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, semantic_dec, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, semantic_dec, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior, semantic_dec = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, semantic_dec, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, semantic_dec, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            if self.stage2 == False:
                self.ema_encoder(self.encoder)
                self.ema_quant_layer(self.quant_layer)
                self.ema_semantic_decoder(self.semantic_decoder)
            self.ema_decoder(self.decoder)
            self.ema_post_quant_layer(self.post_quant_layer)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.semantic_decoder.parameters())+
                                  list(self.quant_layer.parameters())+
                                  list(self.post_quant_layer.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        if self.stage2:
            opt_ae = torch.optim.Adam(list(self.decoder.parameters())+
                                      list(self.semantic_decoder.parameters())+
                                    list(self.post_quant_layer.parameters()),
                                    lr=lr, betas=(0.5, 0.9))            
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior, _ = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"], _ = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


