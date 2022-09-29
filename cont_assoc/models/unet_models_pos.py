import numpy as np
import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
import cont_assoc.models.unet_blocks as blocks
import cont_assoc.utils.predict as pred
import cont_assoc.utils.testing as testing
import cont_assoc.utils.save_features_cunet as sf                # modified
from cont_assoc.utils.evaluate_panoptic import PanopticKittiEvaluator
from cont_assoc.utils.evaluate_4dpanoptic import PanopticKitti4DEvaluator
from utils.common_utils import SemKITTI2train
from cont_assoc.models.loss_contrastive import SupConLoss
from cont_assoc.utils.assoc_module_with_pos import AssociationModule
import cont_assoc.utils.contrastive as cont
import torch.optim.lr_scheduler as schedul
import MinkowskiEngine as ME


class UNet(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.ignore_label = cfg.DATA_CONFIG.DATALOADER.CONVERT_IGNORE_LABEL         

        self.voxel_feature_extractor = blocks.VoxelFeatureExtractor(cfg)          

        self.pos_unet = PosCUnet(cfg)
        self.ins_loss = SupConLoss(temperature=0.1)        
        self.val_loss = np.float(0)
        self.val_num = 0
        self.evaluator4D = PanopticKitti4DEvaluator(cfg=cfg)     
        feat_size = cfg.DATA_CONFIG.DATALOADER.POS_DATA_DIM
        self.pos_enc = cont.PositionalEncoder(max_freq=100000,
                                              feat_size=feat_size,
                                              dimensionality=3)
        weights = cfg.TRACKING.ASSOCIATION_WEIGHTS
        thresholds = cfg.TRACKING.ASSOCIATION_THRESHOLDS
        use_poses = cfg.MODEL.USE_POSES
        self.AssocModule = AssociationModule(weights, thresholds, self.pos_unet,
                                             self.pos_enc, use_poses, self.voxel_feature_extractor, cfg)


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


    def get_ins_feat(self, x, ins_pred, raw_features):          
        #Group points into instances
        pt_raw_feat = pred.feat_voxel2point(raw_features,x)     
        pt_coordinates = x['pt_cart_xyz']
        grid_pos = x['grid']

        # coordinates, features, n_instances, ins_ids, ins_pred = cont.group_instances(pt_coordinates, pt_raw_feat, ins_pred)
        coordinates, features, n_instances, ins_ids, ins_pred, grid_coordinates = self.group_instances_with_grid(pt_coordinates, pt_raw_feat, ins_pred, grid_pos)
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


    def get_pq(self):
        return self.evaluator.get_mean_pq()

    def merge_predictions(self, x, sem_logits, pred_offsets, pt_ins_feat):
        #Sem labels
        pt_sem_pred = pred.sem_voxel2point(sem_logits, x)

        #Ins labels
        last_ins_id = 0
        clust_bandwidth = self.cfg.MODEL.POST_PROCESSING.BANDWIDTH
        ins_embedding_dim = self.cfg.MODEL.INS_HEAD.EMBEDDING_CHANNEL
        ins_pred = pred.cluster_ins(pt_sem_pred, pt_ins_feat, pred_offsets, x,
                                    clust_bandwidth, last_ins_id)
        #Majority voting
        sem_pred = pred.majority_voting(pt_sem_pred, ins_pred)

        return sem_pred, ins_pred


    def merge_pos(self, pt_coors, pt_features):
        for i in range(len(pt_features)):
            pos_encoding = self.pos_enc(pt_coors[i])
            pt_features[i] = pt_features[i] + pos_encoding
        return pt_features

        #create sparse tensor
        # all_feat = [item for sublist in pt_features for item in sublist]
        # all_coors = [item for sublist in pt_coors for item in sublist]
        # c_, f_ = ME.utils.sparse_collate(all_coors, all_feat, dtype=torch.float32)
        # sparse = ME.SparseTensor(features=f_, coordinates=c_.int(),device='cuda')
        # return sparse



    def forward(self, x):
        
        coordinates, voxel_features = self.voxel_feature_extractor(x)
        voxel_features_pos = self.merge_pos([coordinates[:,1:]], [voxel_features])[0]
        batch_size = len(x['grid'])
        ins_features = self.pos_unet(coordinates, voxel_features_pos, batch_size)                    

        return ins_features

    #############################################################

    def configure_optimizers(self):  
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                self.parameters()), lr=self.cfg.TRAIN.LR)
        eta_min=self.cfg.TRAIN.LR/self.cfg.TRAIN.SCHEDULER.DIV_FACTOR
        scheduler = schedul.CosineAnnealingLR(optimizer,
                                              self.cfg.TRAIN.MAX_EPOCH,
                                              eta_min=eta_min)
        return [optimizer], [scheduler]

    def sample_loss(self, norm_features, pos_labels, sem_labels, pos_scans):
        _feats = []
        _pos_labels = []
        _sem_labels = []
        ids, n_ids = torch.unique(pos_labels,return_counts=True)
        sorted_n_ids, indices = torch.sort(n_ids, descending=True)
        ids = ids[indices]
        ins_num = 0
        for i in range(len(ids)): #iterate over all instances
            if ins_num > 10000:
                break
            if sorted_n_ids[i] < 1000: #filter too small instances 30
                ins_num += sorted_n_ids[i]
                pt_idx = torch.where(pos_labels==ids[i])[0]
                feat = norm_features[pt_idx]
                s_labels = sem_labels[pt_idx]
                p_labels = pos_labels[pt_idx]
                _pos_labels.append(p_labels)
                _feats.append(feat)
                _sem_labels.append(s_labels)
            elif sorted_n_ids[i] >= 1000 and sorted_n_ids[i] <= 2000:
                pick_num = int(sorted_n_ids[i]*0.5)
                ins_num += pick_num
                pt_idx = torch.where(pos_labels==ids[i])[0]
                pt_idx = pt_idx[torch.randperm(sorted_n_ids[i])][:pick_num]
                feat = norm_features[pt_idx]
                s_labels = sem_labels[pt_idx]
                p_labels = pos_labels[pt_idx]
                _pos_labels.append(p_labels)
                _feats.append(feat)
                _sem_labels.append(s_labels)
            elif sorted_n_ids[i] > 2000:
                pick_num = 1000
                ins_num += pick_num
                pt_idx = torch.where(pos_labels==ids[i])[0]
                pt_idx = pt_idx[torch.randperm(sorted_n_ids[i])][:pick_num]
                feat = norm_features[pt_idx]
                s_labels = sem_labels[pt_idx]
                p_labels = pos_labels[pt_idx]
                _pos_labels.append(p_labels)
                _feats.append(feat)
                _sem_labels.append(s_labels)

        features = torch.cat([i for i in _feats])
        pos_labels = torch.cat([i for i in _pos_labels])
        sem_labels = torch.cat([i for i in _sem_labels])

        return features, pos_labels, sem_labels

    def getLoss(self, x, features):
        loss = {}
        sem_labels = [torch.from_numpy(i).type(torch.LongTensor).cuda()
                      for i in x['pt_labs']]
        sem_labels = (torch.cat([i for i in sem_labels])) 
        pos_labels = [torch.from_numpy(i).type(torch.LongTensor).cuda()
                      for i in x['pt_ins_labels']]
        pos_labels = (torch.cat([i for i in pos_labels])) 
        pt_raw_feat = pred.feat_voxel2point(features, x)
        pt_raw_feat = (torch.cat([i for i in pt_raw_feat])).cuda()            
        norm_features = F.normalize(pt_raw_feat)

        valid = x['pt_valid']
        pos_scans = x['pos_scans'][0]
        valid = (np.concatenate([i for i in valid]))
        pos_labels = pos_labels[valid]
        sem_labels = sem_labels[valid]
        norm_features = norm_features[valid]
        # different number of samples for loss function
        feats, pos_l, sem_l = self.sample_loss(norm_features, pos_labels, sem_labels, pos_scans)
        ins_loss = self.ins_loss(feats, pos_l, sem_l)         # torch.Size([13242, 128])
        loss['unet_loss'] = ins_loss
        return loss


    def training_step(self, batch, batch_idx):

        x = batch
        instance_feat = self(x)

        loss = self.getLoss(x, instance_feat)
        # self.log('training_loss', loss['unet_loss'])
        torch.cuda.empty_cache()

        return loss['unet_loss']

    
    def validation_step(self, batch, batch_idx):
        x = batch
        instance_features = self(x)
        loss = self.getLoss(x, instance_features)
        self.val_loss += loss['unet_loss']
        self.val_num += 1
        self.log('val_loss', loss['unet_loss'])
        torch.cuda.empty_cache()

        return

        #load predictions for the whole scan (merge_predictions)
        # ins_pred = x['pt_ins_pred']             # (123159,)
        # sem_pred = x['pt_sem_pred']             # (123159,)

        # ins_feat, n_ins, ins_ids, ins_pred, tr_input = self.get_ins_feat(x, ins_pred, instance_feat)
        # n_instances = n_ins
        # ins_feat = torch.split(ins_feat, n_instances)
        # points = tr_input['pt_coors']
        # features = tr_input['pt_features']
        # grid_coors = tr_input['grid_coors']
        # poses = x['pose']
        # poses = [[p] for p in poses]
        # ins_pread = self.AssocModule.associate(ins_pred, ins_feat, points, features, poses, ins_ids, grid_coors, x)
        # self.evaluator4D.update(sem_pred, ins_pred, x)

        # torch.cuda.empty_cache()
        # return

        


    def validation_epoch_end(self, outputs):
        val_loss = self.val_loss / self.val_num 
        self.log('val_loss',val_loss)
        self.val_loss = 0
        self.val_num = 0

        # self.evaluator4D.calculate_metrics()
        # AQ = self.evaluator4D.get_mean_aq()
        # self.log('AQ',AQ)

        # self.AssocModule.clear()
        # self.evaluator4D.clear()
        

# Modules

class CylinderEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        output_shape = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE
        num_input_features = cfg.MODEL.VOXEL_FEATURES.FEATURE_DIM
        self.nclasses = cfg.DATA_CONFIG.NCLASS
        self.n_height = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE[2]
        init_size = cfg.MODEL.BACKBONE.INIT_SIZE

        sparse_shape = np.array(output_shape)
        self.sparse_shape = sparse_shape

        self.contextBlock = blocks.ResBlock(num_input_features, init_size, indice_key="context")
        self.downBlock0 = blocks.DownResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down0")
        self.downBlock1 = blocks.DownResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down1")
        self.downBlock2 = blocks.DownResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False, indice_key="down2")
        self.downBlock3 = blocks.DownResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False, indice_key="down3")

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        x = self.contextBlock(x)
        down0_feat, down0_skip = self.downBlock0(x)
        down1_feat, down1_skip = self.downBlock1(down0_feat)
        down2_feat, down2_skip = self.downBlock2(down1_feat)
        down3_feat, down3_skip = self.downBlock3(down2_feat)

        skips = [down0_skip, down1_skip, down2_skip, down3_skip]

        return down3_feat, skips

class CylinderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        init_size = cfg.MODEL.BACKBONE.INIT_SIZE

        self.upBlock0 = blocks.UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down3")
        self.upBlock1 = blocks.UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down2")
        self.upBlock2 = blocks.UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down1")
        self.upBlock3 = blocks.UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down0")

        self.decompBlock = blocks.DimDecBlock(2*init_size, 2*init_size, indice_key="decomp")

    def forward(self, x, skips):

        up0 = self.upBlock0(x, skips[3])
        up1 = self.upBlock1(up0, skips[2])
        up2 = self.upBlock2(up1, skips[1])
        up3 = self.upBlock3(up2, skips[0])

        upsampled_feat = self.decompBlock(up3)

        upsampled_feat.features = torch.cat((upsampled_feat.features, up3.features), 1)

        return upsampled_feat, upsampled_feat

class CylinderSemanticHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nclasses = cfg.DATA_CONFIG.NCLASS
        init_size = cfg.MODEL.BACKBONE.INIT_SIZE

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, sem_fea):
        logits = self.logits(sem_fea)
        return logits

class CylinderInstanceHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        init_size = cfg.MODEL.BACKBONE.INIT_SIZE

        self.pt_fea_dim = 4 * init_size
        self.out_dim = 3 #offset x,y,z

        self.conv1 = blocks.conv3x3(self.pt_fea_dim, self.pt_fea_dim, indice_key='offset_head_conv1')
        self.bn1 = nn.BatchNorm1d(self.pt_fea_dim)
        self.act1 = nn.LeakyReLU()
        self.conv2 = blocks.conv3x3(self.pt_fea_dim, 2 * init_size, indice_key='offset_head_conv2')
        self.bn2 = nn.BatchNorm1d(2 * init_size)
        self.act2 = nn.LeakyReLU()
        self.conv3 = blocks.conv3x3(2 * init_size, init_size, indice_key='offset_head_conv3')
        self.bn3 = nn.BatchNorm1d(init_size)
        self.act3 = nn.LeakyReLU()

        self.offset = nn.Sequential(
            nn.Linear(init_size+3, init_size, bias=True),
            nn.BatchNorm1d(init_size),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(init_size, self.out_dim, bias=True)

    def forward(self, fea, x):
        out = self.conv1(fea)
        out.features = self.act1(self.bn1(out.features))
        out = self.conv2(out)
        out.features = self.act2(self.bn2(out.features))
        out = self.conv3(out)
        out.features = self.act3(self.bn3(out.features))

        grid_ind = x['grid']
        xyz = x['pt_cart_xyz']
        out = out.dense()
        out = out.permute(0, 2, 3, 4, 1)
        pt_ins_fea_list = []
        for batch_i, grid_ind_i in enumerate(grid_ind):
            pt_ins_fea_list.append(out[batch_i, grid_ind[batch_i][:,0], grid_ind[batch_i][:,1], grid_ind[batch_i][:,2]])
        pt_pred_offsets_list = []
        for batch_i, pt_ins_fea in enumerate(pt_ins_fea_list):
            pt_pred_offsets_list.append(self.offset_linear(self.offset(torch.cat([pt_ins_fea,torch.from_numpy(xyz[batch_i]).cuda()],dim=1))))

        return pt_pred_offsets_list, pt_ins_fea_list


# Modules
class PosCUnet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.voxel_feature_extractor = blocks.VoxelFeatureExtractor(cfg)          #mm
        self.encoder = CylinderEncoder(cfg)
        self.decoder = CylinderDecoder(cfg)
      
    def forward(self, coordinates, voxel_features,  batch_size):
        encoding, skips = self.encoder(voxel_features, coordinates, batch_size) ## modify
        semantic_feat, instance_feat = self.decoder(encoding, skips)
        return instance_feat


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

