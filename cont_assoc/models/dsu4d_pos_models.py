# import pdb
# pdb.set_trace()
import numpy as np
import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from pytorch_lightning.core.lightning import LightningModule
import cont_assoc.models.unet_blocks as blocks
import cont_assoc.models.panoptic_models as p_models
import cont_assoc.models.unet_models_pos_encoder as u_models
import cont_assoc.utils.predict as pred
import cont_assoc.utils.contrastive as cont
import cont_assoc.utils.testing as testing
from cont_assoc.utils.evaluate_panoptic import PanopticKittiEvaluator
from cont_assoc.utils.evaluate_4dpanoptic import PanopticKitti4DEvaluator
# from cont_assoc.utils.assoc_module import AssociationModule
import cont_assoc.utils.save_features as sf


class DSU4D(LightningModule):
    def __init__(self, ps_cfg, u_cfg):
        super().__init__()
        self.ps_cfg = ps_cfg
        self.u_cfg = u_cfg

        self.panoptic_model = p_models.PanopticCylinder(ps_cfg)
        self.unet_model = u_models.UNet(u_cfg)
        # self.encoder = SparseEncoder(u_cfg)
        self.evaluator4D = PanopticKitti4DEvaluator(cfg=ps_cfg)
        # feat_size = u_cfg.DATA_CONFIG.DATALOADER.POS_DATA_DIM
        # self.pos_enc = cont.PositionalEncoder(max_freq=100000,
                                            #   feat_size=feat_size,
                                            #   dimensionality=3)
        # weights = u_cfg.TRACKING.ASSOCIATION_WEIGHTS
        # thresholds = u_cfg.TRACKING.ASSOCIATION_THRESHOLDS
        # use_poses = u_cfg.MODEL.USE_POSES
        # self.AssocModule = AssociationModule(weights, thresholds, self.encoder,
        #                                      self.pos_enc, use_poses)
        self.last_ins_id = 0

    def load_state_dicts(self, ps_dict, u_dict):
        self.unet_model.load_state_dict(u_dict)
        self.panoptic_model.load_state_dict(ps_dict)

    def merge_predictions(self, x, sem_logits, pred_offsets, pt_ins_feat):
        pt_sem_pred = pred.sem_voxel2point(sem_logits, x)
        clust_bandwidth = self.ps_cfg.MODEL.POST_PROCESSING.BANDWIDTH
        ins_pred = pred.cluster_ins(pt_sem_pred, pt_ins_feat, pred_offsets, x,
                                    clust_bandwidth, self.last_ins_id)
        sem_pred = pred.majority_voting(pt_sem_pred, ins_pred)

        return sem_pred, ins_pred
    
    def get_mean(self, features):
        ins_feats = []
        for i in range(len(features)):
            for j in range(len(features[i])):
                ins_feats.append(torch.mean(features[i][j], 0))
        ins_feat = torch.stack(ins_feats, 0)
        return ins_feat

    
    def group_instances_with_grid(self, pt_coordinates, pt_raw_feat, ins_pred, grid_position):
        coordinates = []
        features = []
        n_instances = []
        ins_ids = []
        grid_coordinates = []
        for i in range(len(pt_coordinates)):#for every scan in the batch
            _coors = []
            _grid_coor = []
            _feats = []
            _ids = []
            pt_coors = pt_coordinates[i]#get point coordinates
            feat = pt_raw_feat[i].numpy()#get point features
            grid_pos = grid_position[i]
            #get instance ids
            pt_ins_id = ins_pred[i]
            valid = pt_ins_id != 0 #ignore id=0
            ids, n_ids = np.unique(pt_ins_id[valid],return_counts=True)
            n_ins = 0
            for ii in range(len(ids)):#iterate over all instances
                if n_ids[ii] > 30:#filter too small instances
                    pt_idx = np.where(pt_ins_id==ids[ii])[0]
                    coors = torch.tensor(pt_coors[pt_idx],device='cuda')
                    grid_coors = torch.tensor(grid_pos[pt_idx],device='cuda')
                    feats = torch.tensor(feat[pt_idx],device='cuda')
                    _coors.extend([coors])
                    _grid_coor.extend([grid_coors])
                    _feats.extend([feats])
                    _ids.extend([ids[ii]])
                    n_ins += 1
            coordinates.append(_coors)
            grid_coordinates.append(_grid_coor)
            features.append(_feats)
            n_instances.append(n_ins)
            ins_ids.append(_ids)
        return coordinates, features, n_instances, ins_ids, ins_pred, grid_coordinates




    def get_ins_feat(self, x, ins_pred, raw_features):          # torch.Size([49315, 128])
        #Group points into instances
        pt_raw_feat = pred.feat_voxel2point(raw_features,x)     # torch.Size([123389, 128])
        pt_coordinates = x['pt_cart_xyz']
        grid_pos = x['grid']

        coordinates, features, n_instances, ins_ids, ins_pred, grid_coordinates= self.group_instances_with_grid(pt_coordinates, pt_raw_feat, ins_pred, grid_pos)

        #Discard scans without instances
        features = [x for x in features if len(x)!=0]
        coordinates = [x for x in coordinates if len(x)!=0]
        grid_coordinates = [x for x in grid_coordinates if len(x)!=0]

        if len(features)==0:#don't run tracking head if no ins
            # return [], [], [], ins_pred
            return [], [], [], ins_pred, {}

        #Get per-instance feature
        tracking_input = {'pt_features':features,'pt_coors':coordinates, 'grid_coors': grid_coordinates}

        # ins_feat = self.unet_model(tracking_input)          
        ins_feat = self.get_mean(features)                      # average the point features to get only one 128-d feature per instance

        if len(coordinates) != len(ins_ids):
            #scans without instances
            new_feats, new_coors, new_grid_coordinates = cont.fix_batches(ins_ids, features, coordinates)       # need to modify
            tracking_input = {'pt_features':new_feats,'pt_coors':new_coors, 'grid_coors': new_grid_coordinates}

        return ins_feat, n_instances, ins_ids, ins_pred, tracking_input

    def track(self, ins_pred, ins_feat, n_instances, ins_ids, tr_input, poses, x):
        #Separate instances of different scans
        points = tr_input['pt_coors']
        features = tr_input['pt_features']
        grid_coors = tr_input['grid_coors']
        ins_feat = torch.split(ins_feat, n_instances)       # torch.Size([7, 128])
        poses = [[p] for p in poses]

        #Instance IDs association
        ins_pred = self.unet_model.AssocModule.associate(ins_pred, ins_feat,
                                                            points, features,
                                                            poses, ins_ids, grid_coors, x)

        self.last_ins_id = self.unet_model.AssocModule.get_last_id()
        self.unet_model.AssocModule.update_last_id(self.last_ins_id)

        return ins_pred

    def forward(self, x):
        sem_logits, pred_offsets, pt_ins_feat, raw_features = self.panoptic_model(x)        # ds-net ps backbone
        sem_pred, ins_pred = self.merge_predictions(x, sem_logits,
                                                    pred_offsets, pt_ins_feat)
        
        pt_raw_feat = pred.feat_voxel2point(raw_features, x)            # N, 128
        pt_raw_feat = [i.numpy() for i in pt_raw_feat]
        # new_feats = np.concatenate(pt_raw_feat, axis=0)
        x['feats'] = pt_raw_feat
        instance_feat = self.unet_model(x)                        # u-net fine tune instance point embedings 128-d

        sem_pred, ins_pred = self.merge_predictions(x, sem_logits, pred_offsets, pt_ins_feat)
        
        ins_feat, n_ins, ins_ids, ins_pred, tracking_input = self.get_ins_feat(x, ins_pred, instance_feat)

        #if no instances, don't track
        if len(ins_feat)!=0:
            ins_pred = self.track(ins_pred, ins_feat, n_ins, ins_ids, tracking_input, x['pose'], x)        # ins_feats [7,128]
        return sem_pred, ins_pred, instance_feat

    def test_step(self, batch, batch_idx):
        x = batch
        sem_pred, ins_pred, raw_features= self(x)

        if 'RESULTS_DIR' in self.ps_cfg:
            results_dir = self.ps_cfg.RESULTS_DIR
            class_inv_lut = self.panoptic_model.evaluator.get_class_inv_lut()
            testing.save_results(sem_pred, ins_pred, results_dir, x, class_inv_lut)

        if 'UPDATE_METRICS' in self.ps_cfg:
            self.panoptic_model.evaluator.update(sem_pred, ins_pred, x)
            self.evaluator4D.update(sem_pred, ins_pred, x)

        if 'SAVE_VAL_PRED' in self.ps_cfg:
            pt_raw_feat = pred.feat_voxel2point(raw_features,x)
            sf.save_features(x, pt_raw_feat, sem_pred, ins_pred, save_preds=True)


# Modules
class SparseEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.DATA_CONFIG.DATALOADER.DATA_DIM #128
        channels = [x * input_dim for x in cfg.MODEL.ENCODER.CHANNELS] #128, 128, 256, 512
        kernel_size = 3

        self.conv1 = SparseConvBlock(
            channels[0],
            channels[1],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv2 = SparseConvBlock(
            channels[1],
            channels[2],
            kernel_size=kernel_size,
            stride=2,
        )

        self.conv3 = SparseConvBlock(
            channels[2],
            channels[3],
            kernel_size=kernel_size,
            stride=2,
        )

        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

        self.final = nn.Sequential(
            SparseLinearBlock(channels[-1], 2*channels[-1]),
            ME.MinkowskiDropout(),
            SparseLinearBlock(2*channels[-1], channels[-1]),
            ME.MinkowskiLinear(channels[-1], channels[-1], bias=True),
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.global_avg_pool(y)
        return self.final(y).F

class SparseLinearBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
                        ME.MinkowskiLinear(in_channel, out_channel, bias=False),
                        ME.MinkowskiBatchNorm(out_channel),
                        ME.MinkowskiLeakyReLU(),
                    )

    def forward(self, x):
        return self.layer(x)

class SparseConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, dimension=3):
        super().__init__()
        self.layer =  nn.Sequential(
                        ME.MinkowskiConvolution(
                            in_channel,
                            out_channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            dimension=dimension),
                        ME.MinkowskiBatchNorm(out_channel),
                        ME.MinkowskiLeakyReLU(),
                    )

    def forward(self, x):
        return self.layer(x)
