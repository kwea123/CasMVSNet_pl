import os
from opt import get_opts
import torch

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets.dtu import DTUDataset

# models
from models.mvsnet import MVSNet
from inplace_abn import InPlaceABN, InPlaceABNSync

from torchvision import transforms as T

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

torch.backends.cudnn.benchmark = True # this increases training speed by 5x

class MVSSystem(pl.LightningModule):
    def __init__(self, hparams):
        super(MVSSystem, self).__init__()
        self.hparams = hparams
        # to unnormalize image for visualization
        self.unpreprocess = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                        std=[1/0.229, 1/0.224, 1/0.225])

        self.loss = loss_dict[hparams.loss_type](ohem=True, topk=0.6)

    def forward(self, imgs, proj_mats, depth_values):
        return self.model(imgs, proj_mats, depth_values)

    def training_step(self, batch, batch_nb):
        imgs, proj_mats, depth_gt, depth_values, mask = batch
        depth_pred, confidence = self.forward(imgs, proj_mats, depth_values)
        loss = self.loss(depth_pred, depth_gt, mask)
        
        with torch.no_grad():
            if batch_nb == 0:
                img_ = self.unpreprocess(imgs[0,0,:,::4,::4]).cpu() # batch 0, ref image, 1/4 scale
                depth_gt_ = visualize_depth(depth_gt[0])
                depth_pred_ = visualize_depth(depth_pred[0]*mask[0])
                prob = visualize_prob(confidence[0]*mask[0])
                stack = torch.stack([img_, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
                self.logger.experiment.add_images('train/image_GT_pred_prob',
                                                  stack, self.global_step)

            abs_err = abs_error(depth_pred, depth_gt, mask).mean()
            acc_1mm = acc_threshold(depth_pred, depth_gt, mask, 1).mean()
            acc_2mm = acc_threshold(depth_pred, depth_gt, mask, 2).mean()
            acc_4mm = acc_threshold(depth_pred, depth_gt, mask, 4).mean()

        return {'loss': loss,
                'progress_bar': {'train_abs_err': abs_err},
                'log': {'train/loss': loss,
                        'train/abs_err': abs_err,
                        'train/acc_1mm': acc_1mm,
                        'train/acc_2mm': acc_2mm,
                        'train/acc_4mm': acc_4mm,
                        'lr': get_learning_rate(self.optimizer)}
               }

    def validation_step(self, batch, batch_nb):
        imgs, proj_mats, depth_gt, depth_values, mask = batch
        depth_pred, confidence = self.forward(imgs, proj_mats, depth_values)
        loss = self.loss(depth_pred, depth_gt, mask)

        with torch.no_grad():
            if batch_nb == 0:
                img_ = self.unpreprocess(imgs[0,0,:,::4,::4]).cpu() # batch 0, ref image, 1/4 scale
                depth_gt_ = visualize_depth(depth_gt[0])
                depth_pred_ = visualize_depth(depth_pred[0]*mask[0])
                prob = visualize_prob(confidence[0]*mask[0])
                stack = torch.stack([img_, depth_gt_, depth_pred_, prob]) # (4, 3, H, W)
                self.logger.experiment.add_images('val/image_GT_pred_prob',
                                                  stack, self.global_step)

            abs_err = abs_error(depth_pred, depth_gt, mask)
            acc_1mm = acc_threshold(depth_pred, depth_gt, mask, 1)
            acc_2mm = acc_threshold(depth_pred, depth_gt, mask, 2)
            acc_4mm = acc_threshold(depth_pred, depth_gt, mask, 4)

        return {'val_loss': loss,
                'val_abs_err': abs_err,
                'val_acc_1mm': acc_1mm,
                'val_acc_2mm': acc_2mm,
                'val_acc_4mm': acc_4mm,
                }

    def validation_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_abs_err = torch.cat([x['val_abs_err'] for x in outputs]).mean()
        mean_acc_1mm = torch.cat([x['val_acc_1mm'] for x in outputs]).mean()
        mean_acc_2mm = torch.cat([x['val_acc_2mm'] for x in outputs]).mean()
        mean_acc_4mm = torch.cat([x['val_acc_4mm'] for x in outputs]).mean()

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_abs_err': mean_abs_err},
                'log': {'val/loss': mean_loss,
                        'val/abs_err': mean_abs_err,
                        'val/acc_1mm': mean_acc_1mm,
                        'val/acc_2mm': mean_acc_2mm,
                        'val/acc_4mm': mean_acc_4mm,
                        }
               }

    def configure_optimizers(self):
        if self.hparams.use_syncbn and self.hparams.num_gpus>1:
            norm_act = InPlaceABNSync
        else:
            norm_act = InPlaceABN

        self.model = MVSNet(norm_act).cuda()

        # if num gpu is 1, print model structure and number of params
        if self.hparams.num_gpus == 1:
            # print(self.model)
            print('number of parameters : %.2f M' % 
                  (sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6))
        
        # load model if checkpoint path is provided
        if self.hparams.ckpt_path != '':
            print('Load model from', self.hparams.ckpt_path)
            load_ckpt(self.model, self.hparams.ckpt_path, self.hparams.prefixes_to_ignore)

        self.optimizer = get_optimizer(self.hparams, self.model)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        train_dataset = DTUDataset(root_dir=self.hparams.root_dir,
                                   split='train',
                                   n_views=self.hparams.n_views,
                                   n_depths=self.hparams.n_depths,
                                   interval_scale=self.hparams.interval_scale)
        if self.hparams.num_gpus > 1:
            sampler = DistributedSampler(train_dataset)
        else:
            sampler = None
        return DataLoader(train_dataset, 
                          shuffle=(sampler is None),
                          sampler=sampler,
                          num_workers=8,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        val_dataset = DTUDataset(root_dir=self.hparams.root_dir,
                                 split='val',
                                 n_views=self.hparams.n_views,
                                 n_depths=self.hparams.n_depths,
                                 interval_scale=self.hparams.interval_scale)
        if self.hparams.num_gpus > 1:
            sampler = DistributedSampler(val_dataset)
        else:
            sampler = None
        return DataLoader(val_dataset, 
                          shuffle=(sampler is None),
                          sampler=sampler,
                          num_workers=8,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

if __name__ == '__main__':
    hparams = get_opts()
    system = MVSSystem(hparams)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join('ckpts', 
                                                   f'{hparams.exp_name}'),
                                          monitor='val/loss',
                                          mode='min',
                                          save_top_k=1,)

    logger = TestTubeLogger(
        save_dir="logs",
        name=f'{hparams.exp_name}',
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      logger=logger,
                      early_stop_callback=None,
                      weights_summary=None,
                      gpus=hparams.num_gpus,
                      distributed_backend='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=0 if hparams.num_gpus>1 else 5,
                      use_amp=hparams.use_amp,
                      amp_level='O1')

    trainer.fit(system)