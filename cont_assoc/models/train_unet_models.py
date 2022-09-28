# import pdb
# pdb.set_trace()
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import instance_losses
from .loss import lovasz_losses
from pytorch_lightning.core.lightning import LightningModule
import cont_assoc.models.unet_blocks as blocks
import cont_assoc.utils.predict as pred
import cont_assoc.utils.testing as testing
import cont_assoc.utils.save_features_unet as sf                # modified
from cont_assoc.utils.evaluate_panoptic import PanopticKittiEvaluator
from cont_assoc.utils.evaluate_4dpanoptic import PanopticKitti4DEvaluator
from utils.common_utils import SemKITTI2train
import cont_assoc.utils.contrastive as cont
from cont_assoc.models.loss_contrastive import SupConLoss
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.manifold import TSNE 
# import hdbscan
from sklearn.mixture import BayesianGaussianMixture


class UNet(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.ignore_label = cfg.DATA_CONFIG.DATALOADER.CONVERT_IGNORE_LABEL         # 0

        self.voxel_feature_extractor = blocks.VoxelFeatureExtractor(cfg)
        self.encoder = CylinderEncoder(cfg)
        self.decoder = CylinderDecoder(cfg)
        # self.semantic_head = CylinderSemanticHead(cfg)
        # self.instance_head = CylinderInstanceHead(cfg)
        ############################################
        feat_size = cfg.DATA_CONFIG.DATALOADER.POS_DATA_DIM
        # self.pos_enc = cont.PositionalEncoder(max_freq=100000,
        #                                       feat_size=feat_size,
        #                                       dimensionality=3)


        ###########################################
        # self.evaluator = PanopticKittiEvaluator(cfg=cfg)
        self.ins_loss = SupConLoss(temperature=0.1)          # modify
        self.val_loss = np.float(0)
        self.val_num = 0
        self.linearNet = blocks.LinearNet(cfg)


    # def get_pq(self):
    #     return self.evaluator.get_mean_pq()

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


    def forward(self, x):

        coordinates, voxel_features = self.voxel_feature_extractor(x)   # torch.Size([49315, 16])
        # voxel_features = self.sparse_tensor(coordinates, voxel_features)
        encoding, skips = self.encoder(voxel_features, coordinates, len(x['grid'])) ## torch.Size([4790, 512])

        semantic_feat, instance_feat = self.decoder(encoding, skips)            # torch.Size([49315, 128])
        instance_feat =  self.linearNet(instance_feat.features)
        # semantic_logits = self.semantic_head(semantic_feat)
        # predicted_offsets, pt_ins_feat = self.instance_head(instance_feat, x)

        # return semantic_logits, predicted_offsets, pt_ins_feat, instance_feat
        return instance_feat


    def configure_optimizers(self):  # need to modify?
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                self.parameters()), lr=self.cfg.TRAIN.LR)
        return optimizer

    def group_instances(self, norm_features, pos_labels, sem_labels, pos_scans):
        _feats = []
        _pos_labels = []
        _sem_labels = []
        ids, n_ids = torch.unique(pos_labels,return_counts=True)
        sorted_n_ids, indices = torch.sort(n_ids, descending=True)
        ids = ids[indices]
        ins_num = 0
        for i in range(len(ids)): #iterate over all instances
            if ins_num > 20000:
                break
            if sorted_n_ids[0] < 30*pos_scans:
                pt_idx = torch.where(pos_labels==ids[i])[0]
                feat = norm_features[pt_idx]
                s_labels = sem_labels[pt_idx]
                p_labels = pos_labels[pt_idx]
                _pos_labels.append(p_labels)
                _feats.append(feat)
                _sem_labels.append(s_labels)
                # break
            elif sorted_n_ids[i] >= 30*pos_scans and sorted_n_ids[i] < 1000: #filter too small instances 30
                ins_num += sorted_n_ids[i]
                pt_idx = torch.where(pos_labels==ids[i])[0]
                feat = norm_features[pt_idx]
                s_labels = sem_labels[pt_idx]
                p_labels = pos_labels[pt_idx]
                _pos_labels.append(p_labels)
                _feats.append(feat)
                _sem_labels.append(s_labels)
            elif sorted_n_ids[i] >= 1000 and sorted_n_ids[i] <= 4000:
                pick_num = int(sorted_n_ids[i]*0.8)
                ins_num += pick_num
                pt_idx = torch.where(pos_labels==ids[i])[0]
                pt_idx = pt_idx[torch.randperm(sorted_n_ids[i])][:pick_num]
                feat = norm_features[pt_idx]
                s_labels = sem_labels[pt_idx]
                p_labels = pos_labels[pt_idx]
                _pos_labels.append(p_labels)
                _feats.append(feat)
                _sem_labels.append(s_labels)
            elif sorted_n_ids[i] > 4000 and sorted_n_ids[i] <= 8000:
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
            elif sorted_n_ids[i] > 8000:
                pick_num = 4000
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
        sem_labels = (torch.cat([i for i in sem_labels])) #single tensor    torch.Size([622523, 1])
        pos_labels = [torch.from_numpy(i).type(torch.LongTensor).cuda()
                      for i in x['pt_ins_labels']]
        pos_labels = (torch.cat([i for i in pos_labels])) #single tensor    torch.Size([622523, 1])
        # norm_features = F.normalize(features)
        pt_raw_feat = pred.feat_voxel2point_unet(features, x)
        pt_raw_feat = (torch.cat([i for i in pt_raw_feat])).cuda()                # torch.Size([622523, 128])
        norm_features = F.normalize(pt_raw_feat)
        ## clear ins == 0
        valid = x['pt_valid']
        pos_scans = x['pos_scans'][0]
        valid = (np.concatenate([i for i in valid]))
        pos_labels = pos_labels[valid]
        sem_labels = sem_labels[valid]
        norm_features = norm_features[valid]
        # idx = torch.nonzero(pos_labels)
        feats, pos_l, sem_l = self.group_instances(norm_features, pos_labels, sem_labels, pos_scans)
        ins_loss = self.ins_loss(feats, pos_l, sem_l)         # torch.Size([13242, 128])
        loss['unet_loss'] = ins_loss
        return loss


    def training_step(self, batch, batch_idx):

        x = batch
        # semantic_logits, predicted_offsets, pt_ins_feat, instance_feat = self(x)
        instance_feat = self(x)

        loss = self.getLoss(x, instance_feat)
        self.log('training_loss', loss['unet_loss'])
        torch.cuda.empty_cache()

        return loss['unet_loss']

    
    def validation_step(self, batch, batch_idx):
        x = batch
        instance_features = self(x)
        # pt_raw_feat = pred.feat_voxel2point_unet(instance_features, x)[0]    # torch.Size([123389, 128])
        loss = self.getLoss(x, instance_features)
        self.val_loss += loss['unet_loss']
        self.val_num += 1
        self.log('val_loss', loss['unet_loss'])
        torch.cuda.empty_cache()
        # sem_pred, ins_pred = self.merge_predictions(x, sem_logits, pred_offsets, pt_ins_feat)
        # self.evaluator.update(sem_pred, ins_pred, x)
        # valid = x['pt_valid']
        # ins_pred = x['pt_ins_pred']
        # sem_pred = x['pt_sem_pred']
        # ins_ids = x['ids']
        # n_instances = [len(item) for item in x['ids']]
        # pt_raw_feat = pt_raw_feat[valid]                # torch.Size([1715, 128])
        # ins_feat = torch.split(pt_raw_feat, n_instances)  # torch.Size([49315, 128]) # torch.Size([123389, 128])
        # # batched_ins_feat = torch.split(instance_features, n_instances)
        # points = x['pt_coors']
        # features = x['pt_features']
        # poses = x['pose']
        # ins_pread = self.AssocModule.associate(ins_pred, ins_feat, points,
        #                                        features, poses, ins_ids)
        # self.evaluator4D.update(sem_pred, ins_pred, x)

        


    def validation_epoch_end(self, outputs):
        val_loss = self.val_loss / self.val_num 
        self.val_loss = 0
        self.val_num = 0
        self.log('val_loss',val_loss)
        

    def test_step(self, batch, batch_idx):
        x = batch
        sem_logits, pred_offsets, pt_ins_feat, instance_feat = self(x)
        instance_feat = self(x)
        raw_features = instance_feat
        # sem_logits = pred_offsets = pt_ins_feat = []        # mmmm
        # sem_pred, ins_pred = self.merge_predictions(x, sem_logits,
        #                                             pred_offsets, pt_ins_feat)
        sem_pred, ins_pred = [], []
        # pt_raw_feat = pred.feat_voxel2point_unet(raw_features, x)
        # pt_raw_feat = (torch.cat([i for i in pt_raw_feat])).cuda()                # torch.Size([622523, 128])
        
        # # norm_features = F.normalize(pt_raw_feat)
        # norm_features = pt_raw_feat
        
        # valid = x['pt_valid']
        # valid = (np.concatenate([i for i in valid]))
        # norm_features = norm_features[valid].cpu().numpy()
        # pt_raw_feat = pt_raw_feat[valid]
        # pos_labels = [torch.from_numpy(i).type(torch.LongTensor).cuda()
        #               for i in x['pt_ins_labels']]
        # pos_labels = (torch.cat([i for i in pos_labels])) #single tensor    torch.Size([622523, 1])
        # labels_true = np.squeeze(pos_labels[valid].cpu().numpy())
        # # labels_true = labels_true[valid]
        # pt_ins_pred = [torch.from_numpy(i).type(torch.LongTensor).cuda()
        #               for i in x['pt_ins_pred']]
        # pt_ins_pred = (torch.cat([i for i in pt_ins_pred]))
        # pt_ins_pred = pt_ins_pred[valid].cpu().numpy()

        # # tsne = TSNE(n_components=3, init='pca', random_state=501)
        # # tsne_feats = tsne.fit_transform(norm_features)
        # N = len(np.unique(pt_ins_pred))
        # dpgmm = BayesianGaussianMixture(n_components= 15, random_state=42).fit(norm_features)
        # # hdb = hdbscan.HDBSCAN(min_cluster_size=30).fit(norm_features)
        # # db = DBSCAN(eps=5, min_samples=3).fit(norm_features)
        # dpgmm.predict(norm_features)

        # km = KMeans(n_clusters=15, random_state=0).fit(norm_features)
        # labels = km.labels_

        # labels = db.labels_
        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)  # 设置一个样本个数长度的全false向量
        # core_samples_mask[db.core_sample_indices_] = True #将核心样本部分设置为true
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # # print('估计的聚类个数为: %d' % n_clusters_)
        # print("同质性: %0.3f" % metrics.homogeneity_score(labels_true, labels))  # 每个群集只包含单个类的成员。
        # print("完整性: %0.3f" % metrics.completeness_score(labels_true, labels))  # 给定类的所有成员都分配给同一个群集。
        # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))  # 同质性和完整性的调和平均
        # print("调整兰德指数: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
        # print("调整互信息: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
        # print("轮廓系数: %0.3f" % metrics.silhouette_score(tsne_feats, labels))

        if 'RESULTS_DIR' in self.cfg:
            results_dir = self.cfg.RESULTS_DIR
            class_inv_lut = self.evaluator.get_class_inv_lut()
            testing.save_results(sem_pred, ins_pred, results_dir, x, class_inv_lut)

        if 'UPDATE_METRICS' in self.cfg:
            self.evaluator.update(sem_pred, ins_pred, x)

        if 'SAVE_FEATURES' in self.cfg:
            pt_raw_feat = pred.feat_voxel2point_unet(raw_features,x)
            sf.save_features(x, pt_raw_feat, sem_pred, ins_pred, save_preds=False)

        if 'SAVE_VAL_PRED' in self.cfg:
            pt_raw_feat = pred.feat_voxel2point_unet(raw_features,x)
            sf.save_features(x, pt_raw_feat, sem_pred, ins_pred, save_preds=True)

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

