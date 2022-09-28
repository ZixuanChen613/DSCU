import click
import cont_assoc.datasets.unet_dataset as unet_dataset #kitti_dataset
import cont_assoc.models.unet_models_train_aq_loss_1000 as models
from easydict import EasyDict as edict
import os
from os.path import join
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import yaml

def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))

@click.command()
@click.option('--config', '-c', type=str,
              default=join(getDir(__file__), '../config/u_net.yaml'))
@click.option('--weights',
              type=str,
              default=None,
              required=False)


def main(config, weights):    
    cfg = edict(yaml.safe_load(open(config)))

    cfg.DATA_CONFIG.DATALOADER.SHUFFLE = False
    
    data = unet_dataset.SemanticKittiModule(cfg)
    # data.setup()

    model = models.UNet(cfg)
    

    #Load pretrained weights
    if weights:
        pretrain = torch.load(weights, map_location='cpu')
        model.load_state_dict(pretrain['state_dict'],strict=True)

    tb_logger = pl_loggers.TensorBoardLogger('/storage/ukp/work/chen_e/experiments/aq_1000/'+cfg.EXPERIMENT.ID,
                                             default_hp_metric=False)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint = ModelCheckpoint(monitor='AQ',
                                 filename=cfg.EXPERIMENT.ID+'_{epoch:03d}_{AQ:.3f}',
                                 mode='max',
                                 save_last=True)
    #checkpoint = ModelCheckpoint(monitor='val_loss',
     #                            filename=cfg.EXPERIMENT.ID+'_{epoch:03d}_{val_loss:.3f}',
      #                           mode='min',
       #                          save_last=True)


    trainer = Trainer(gpus=1, #cfg.TRAIN.N_GPUS,
                      val_check_interval=0.8,
                      logger=tb_logger,
                      max_epochs= cfg.TRAIN.MAX_EPOCH,
                      log_every_n_steps=10,
                      callbacks=[lr_monitor, checkpoint])


    trainer.fit(model,data)

if __name__ == "__main__":
    main()

