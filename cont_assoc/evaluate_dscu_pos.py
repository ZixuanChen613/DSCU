import click
import cont_assoc.datasets.kitti_dataset as kitti_dataset
import cont_assoc.datasets.unet_dataset as unet_dataset
import cont_assoc.models.dscu_pos_models as models
from easydict import EasyDict as edict
import os
from os.path import join
from pytorch_lightning import Trainer
import subprocess
import torch
import yaml

def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))

@click.command()
@click.option('--test_set', is_flag=True)
@click.option('--ckpt_ps', type=str, required=True)
@click.option('--ckpt_u', type=str, required=True)
@click.option('--save', is_flag=True)
@click.option('--save_val_pred', is_flag=True)
@click.option('--config_ps', '-cs', type=str,
              default=join(getDir(__file__), '../config/panoptic_cylinder.yaml'))
@click.option('--config_u', '-u', type=str,
              default=join(getDir(__file__), '../config/u_net.yaml'))

def main(config_ps, config_u, ckpt_ps, ckpt_u, save, save_val_pred, test_set):
    ps_cfg = edict(yaml.safe_load(open(config_ps)))
    u_cfg = edict(yaml.safe_load(open(config_u)))

    if save:
        results_dir = create_dirs(test_set)
        ps_cfg.RESULTS_DIR = results_dir

    if save_val_pred:
        ps_cfg.SAVE_VAL_PRED = 'True'          # save val pred
        
    ps_cfg.UPDATE_METRICS = 'True'

    ckpt_ps_path = join(getDir(__file__), ckpt_ps)
    checkpoint_ps = torch.load(ckpt_ps_path, map_location='cpu')
    ckpt_u_path = join(getDir(__file__), ckpt_u)
    checkpoint_u = torch.load(ckpt_u_path, map_location='cpu')

    model = models.DSU4D(ps_cfg, u_cfg)
    model.load_state_dicts(checkpoint_ps['state_dict'],checkpoint_u['state_dict'])

    data = kitti_dataset.SemanticKittiModule(ps_cfg)
    data.setup()

    trainer = Trainer(gpus=u_cfg.EVAL.N_GPUS, logger=False)

    print("Setup finished, starting evaluation")

    if test_set:
        trainer.test(model,data.test_dataloader())
    else:
        trainer.test(model,data.val_dataloader())

    model.panoptic_model.evaluator.print_results()
    print("#############################################################")
    model.evaluator4D.calculate_metrics()
    model.evaluator4D.print_results()

def create_dirs(test_set):
    results_dir = join('/_data/dscu_pos', 'output', 'val', 'dsu4d', 'sequences')
    if test_set:
        results_dir = join('/_data/dscu_pos', 'output', 'test', 'dsu4d', 'sequences')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    if test_set:
        for i in range(11,22):
            sub_dir = os.path.join(results_dir, str(i).zfill(2), 'predictions')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir, exist_ok=True)
    else:
        sub_dir = os.path.join(results_dir, str(8).zfill(2), 'predictions')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)
    return results_dir


if __name__ == "__main__":
    main()
