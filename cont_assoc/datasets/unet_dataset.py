# import pdb
# pdb.set_trace()
import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
import pickle
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from tqdm import tqdm
from scipy import stats as s

class SemanticKittiModule(LightningDataModule):
    def __init__(self, cfg, verbose=True):
        super().__init__()
        self.cfg = cfg
        self.verbose = verbose

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        train_split = 'train'
        if 'SAVE_FEATURES' in self.cfg.keys():
            val_split = 'train'
            print("using train split as val split")
        else:
            val_split = 'valid'
        test_split = 'test'

        ########## Point dataset splits
        train_pt_dataset = SemanticKitti(
            self.cfg.DATA_CONFIG.DATASET_PATH + '/sequences/',
            self.cfg.DATA_CONFIG.UNET_TRAIN_PATH + '/sequences/',
            pos_scans = self.cfg.TRAIN.POS_SCANS,       # 2
            split = train_split,
            r_pos_scans = self.cfg.TRAIN.RANDOM_POS_SCANS
        )

        val_pt_dataset = SemanticKitti(
            self.cfg.DATA_CONFIG.DATASET_PATH + '/sequences/',
            self.cfg.DATA_CONFIG.UNET_VAL_PATH + '/sequences/',
            pos_scans = 0,
            split = val_split
        )

        test_pt_dataset = SemanticKitti(
            self.cfg.DATA_CONFIG.DATASET_PATH + '/sequences/',
            self.cfg.DATA_CONFIG.UNET_VAL_PATH + '/sequences/',  ## need to modify
            pos_scans = 0,
            split = test_split
        )
        
        ########## Voxel spherical dataset splits
        self.train_set = CylindricalSemanticKitti(
            train_pt_dataset,
            grid_size = self.cfg.DATA_CONFIG.DATALOADER.GRID_SIZE,
            ignore_label = self.cfg.DATA_CONFIG.DATALOADER.CONVERT_IGNORE_LABEL,
            fixed_volume_space = self.cfg.DATA_CONFIG.DATALOADER.FIXED_VOLUME_SPACE,
        )

        self.val_set = CylindricalSemanticKitti(
            val_pt_dataset,
            grid_size = self.cfg.DATA_CONFIG.DATALOADER.GRID_SIZE,
            ignore_label = self.cfg.DATA_CONFIG.DATALOADER.CONVERT_IGNORE_LABEL,
            fixed_volume_space = self.cfg.DATA_CONFIG.DATALOADER.FIXED_VOLUME_SPACE,
        )

        self.test_set = CylindricalSemanticKitti(
            test_pt_dataset,
            grid_size = self.cfg.DATA_CONFIG.DATALOADER.GRID_SIZE,
            ignore_label = self.cfg.DATA_CONFIG.DATALOADER.CONVERT_IGNORE_LABEL,
            fixed_volume_space = self.cfg.DATA_CONFIG.DATALOADER.FIXED_VOLUME_SPACE,
        )

    def train_dataloader(self):
        self.train_loader = DataLoader(
            dataset = self.train_set,
            batch_size = self.cfg.EVAL.BATCH_SIZE,
            collate_fn = collate_fn_BEV,
            shuffle = self.cfg.DATA_CONFIG.DATALOADER.SHUFFLE,
            num_workers = self.cfg.DATA_CONFIG.DATALOADER.NUM_WORKER,
            pin_memory = True,
            drop_last = False,
            timeout = 0
        )
        self.train_iter = iter(self.train_loader)
        return self.train_loader

    def val_dataloader(self):
        self.valid_loader = DataLoader(
            dataset = self.val_set,
            batch_size = self.cfg.EVAL.BATCH_SIZE,
            collate_fn = collate_fn_BEV,
            shuffle = False,
            num_workers = self.cfg.DATA_CONFIG.DATALOADER.NUM_WORKER,
            pin_memory = True,
            drop_last = False,
            timeout = 0
        )
        self.valid_iter = iter(self.valid_loader)
        return self.valid_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(
            dataset = self.test_set,
            batch_size = self.cfg.EVAL.BATCH_SIZE,
            collate_fn = collate_fn_BEV,
            shuffle = False,
            num_workers = self.cfg.DATA_CONFIG.DATALOADER.NUM_WORKER,
            pin_memory = True,
            drop_last = False,
            timeout = 0
        )
        self.test_iter = iter(self.test_loader)
        return self.test_loader

class SemanticKitti(Dataset):
    def __init__(self, data_path, unet_path, pos_scans, split='train', seq=None, r_pos_scans=False):  # modify unet_path

        self.pos_scans = pos_scans
        self.data_path = data_path
        self.unet_path = unet_path
        self.random_pos_scans = r_pos_scans

        with open("datasets/semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        SemKITTI_label_name = dict()
        for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
            SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]
        self.learning_map = semkittiyaml['learning_map']
        self.split = split
        split = semkittiyaml['split'][self.split]  # 'train' self.split

        self.im_idx = []
        self.unet_idx = []
        pose_files = []
        calib_files = []
        unet_files = []
        empty_files = []

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path,str(i_folder).zfill(2),'velodyne']))
            pose_files.append(absoluteDirPath(data_path+str(i_folder).zfill(2)+'/poses.txt'))
            calib_files.append(absoluteDirPath(data_path+str(i_folder).zfill(2)+'/calib.txt'))

            # empty_files.append(absoluteDirPath(unet_path+str(i_folder).zfill(2)+'/empty.txt'))
            self.unet_idx += absoluteFilePaths('/'.join([self.unet_path,str(i_folder).zfill(2),'scans']))

           
        self.unet_idx.sort()
        self.im_idx.sort()
        self.poses = load_poses(pose_files, calib_files)

        self.things = ['car', 'truck', 'bicycle', 'motorcycle', 'bus', 'person',
                       'bicyclist', 'motorcyclist']
        self.stuff = ['road', 'sidewalk', 'parking', 'other-ground', 'building',
                      'vegetation', 'trunk', 'terrain', 'fence', 'pole', 'traffic-sign']
        self.things_ids = []
        for i in sorted(list(semkittiyaml['labels'].keys())):
            if SemKITTI_label_name[semkittiyaml['learning_map'][i]] in self.things:
                self.things_ids.append(i)

    def __len__(self):
        return len(self.im_idx)

    def __getitem__(self, index):

        raw_data = []
        ref = []
        sem_labels = []
        ins_labels = []
        valid = []
        new_feats = []
        ids = []
        pt_sem_preds = []#for validation predictions
        pt_ins_preds = []#for validation predictions

        if self.split == 'test':
            raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:,0],dtype=int),axis=1)
            sem_labels = annotated_data
            ins_labels = annotated_data
            valid = annotated_data
            ############## modify
        else:
            # raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
            # annotated_data = np.fromfile(self.im_idx[index].replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1))
            # sem_labels = annotated_data & 0xFFFF #delete high 16 digits binary
            # ins_labels = annotated_data
            # valid = np.isin(sem_labels, self.things_ids).reshape(-1) # use 0 to filter out valid indexes is enough
            # sem_labels = np.vectorize(self.learning_map.__getitem__)(sem_labels)

            #####
            # select random scan based on index
            fname = self.unet_idx[index]          # '/_data/zixuan/data_0620/single_frame/validation_predictions/sequences/08/scans/0000000.npy'
            scan = int(fname[-10:-4])           # 0
            seq = fname[-19:-17]                # 08
            scans = [scan]
            pos_idx = [index]
            #select all scans in the sequence
            if self.random_pos_scans:           # True
                #random number [1,pos_scans]
                n_scans = np.random.randint(0,self.pos_scans+1)+1
                #if using previous and future scans
                pair = bool(np.random.randint(0,2))
            else:
                n_scans = self.pos_scans + 1
                pair = True
            for i in range(1,n_scans):
                if pair:
                    if scan-i >= 0:
                        scans.append(scan-i)
                        pos_idx.append(index-i)
                scans.append(scan+i)
                pos_idx.append(index+i)
            scans.sort()            # [0, 1, 2]
            pos_idx.sort()          # [0, 1, 2]

            prev_scan = 0
            if scans[0] > 0:
                prev_scan = scans[0] - 1

            seq_first_pose = []

            for i in range(len(scans)):
                if i == 0:
                    prev_scan_path = absoluteDirPath(self.unet_path+seq+'/scans/'+str(prev_scan).zfill(6)+'.npy')
                    if os.path.exists(prev_scan_path):
                        prev_data = np.load(prev_scan_path,allow_pickle=True)
                        prev_pose = self.poses[prev_scan]
                        prev_ids = prev_data[2]
                        prev_coors = prev_data[5]
                        # prev_coors_T = apply_pose(prev_coors,prev_pose)
                    else:
                        prev_pose = []
                        prev_ids = []
                        prev_coors = []
                        # prev_coors_T = []
                scan_path = absoluteDirPath(self.unet_path+seq+'/scans/'+str(scans[i]).zfill(6)+'.npy')         # '/_data/zixuan/data_0620/single_frame/validation_predictions/sequences/08/scans/000000.npy'
                if os.path.exists(scan_path):
                    #Check max number of points to avoid running out of memory
                    # if sum(n_pts) > 100000: #max n_pts=5e5, bs=4 --> max n_pts=125k  ######## modify
                    #     break
                    pose = self.poses[pos_idx[i]]
                    if len(seq_first_pose) == 0:
                        seq_first_pose = pose
                    
                    bin_path = absoluteDirPath(self.data_path+seq+'/velodyne/'+str(scans[i]).zfill(6)+'.bin')
                    _raw_data = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
                    _ref = _raw_data[:,3]
                    _coors = _raw_data[:,:3]
                    annotated_data = np.fromfile(bin_path.replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1))
                    _sem_labels = annotated_data & 0xFFFF #delete high 16 digits binary
                    _ins_labels = annotated_data
                    _valid = np.isin(_sem_labels, self.things_ids).reshape(-1) # use 0 to filter out valid indexes is enough
                    _sem_labels = np.vectorize(self.learning_map.__getitem__)(_sem_labels)

                    data = np.load(scan_path,allow_pickle=True)
                    
                    # _ids = data[2]
                    # _pos_lab = _ids
                    # _sem_lab = data[3]
                    # _n_pt = data[4]
                    # _coors = data[5]
                    # _feats = data[6]

                    _new_feats = data[7]
                    _ids = data[2]


                    # if self.aug is not None and self.aug.DO_AUG == True:
                    #     _coors, _feats, _n_pt = self.apply_augmentations(_coors, _feats, _n_pt)

                    _coors_T = apply_pose(_coors, pose)

                    #shift coors to local frame of first scan in the sequence
                    if self.split == 'train':
                        _coors = apply_pose(_coors_T, np.linalg.inv(seq_first_pose)).numpy()


                    raw_data.append(_coors)
                    ref = np.append(ref, _ref)
                    sem_labels = np.append(sem_labels, _sem_labels)
                    ins_labels = np.append(ins_labels ,_ins_labels.astype(np.int32))
                    valid = np.append(valid, _valid)
                    new_feats.append(_new_feats)
                    if data.shape[0] == 13:
                        ids.extend(_ids)
                        pt_sem_pred = data[8]
                        pt_ins_pred = data[9]

                    # prev_coors = _coors
                    # prev_coors_T = _coors_T
                    # prev_ids = _ids
        pos_scans = len(scans)
        raw_data = np.concatenate(raw_data, axis=0)
        data_tuple = (raw_data, sem_labels.reshape((-1,1)).astype(np.uint8))
        data_tuple += (ref.astype(np.float32),)#ref
        data_tuple += (ins_labels.reshape((-1,1)).astype(np.int32), valid.astype(np.bool))#ins ids
        data_tuple += (self.im_idx[index],)#filename
        data_tuple += (self.poses[index],)#pose
        new_feats = np.concatenate(new_feats, axis=0)
        data_tuple += (new_feats,) #feats   [m*Ni, 128]
        data_tuple += (pos_scans,)
        data_tuple += (ids,)

        if data.shape[0] == 13:
            data_tuple += (pt_sem_pred,)
            data_tuple += (pt_ins_pred,)
        
        return data_tuple

class CylindricalSemanticKitti(Dataset):
  def __init__(self, in_dataset, grid_size, min_rad=-np.pi/4, max_rad=np.pi/4,
               ignore_label = 255, fixed_volume_space= False,
               max_volume_space = [50,np.pi,1.5], min_volume_space = [3,-np.pi,-3],
               center_type='Axis_center'):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.ignore_label = ignore_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

        self.center_type = center_type

  def __len__(self):
        return len(self.point_cloud_dataset)

  def __getitem__(self, index):
        data = self.point_cloud_dataset[index]
        if len(data) == 9:
            xyz,labels,sig,ins_labels,valid,pcd_fname,feats,pos_scans,ids = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 10:
            xyz,labels,sig,ins_labels,valid,pcd_fname,pose,feats,pos_scans,ids = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 11:
            xyz,labels,sig,ins_labels,valid,pcd_fname,feats,ids,pt_sem_pred,pt_ins_pred = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 12:
            xyz,labels,sig,ins_labels,valid,pcd_fname,pose,feats,pos_scans,ids,pt_sem_pred,pt_ins_pred = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else: raise Exception('Return invalid data tuple')

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:,0],100,axis = 0)
        min_bound_r = np.percentile(xyz_pol[:,0],0,axis = 0)
        max_bound = np.max(xyz_pol[:,1:],axis = 0)
        min_bound = np.min(xyz_pol[:,1:],axis = 0)
        max_bound = np.concatenate(([max_bound_r],max_bound))
        min_bound = np.concatenate(([min_bound_r],min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1) # (size-1) could directly get index starting from 0, very convenient

        if (intervals==0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int) # point-wise grid index

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        # process valid voxel labels
        valid_voxels, vox_to_point = np.unique(grid_ind, return_inverse=True, axis=0)
        voxel_labels = np.ones(valid_voxels.shape[0],dtype = np.uint8)*self.ignore_label

        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,2],grid_ind[:,1],grid_ind[:,0])),:] #same order as coordinates to create sparse tensor when using np.unique

        voxel_labels = nb_get_voxel_labels(np.copy(voxel_labels),label_voxel_pair) #get valid voxel labels
        voxel_labels = voxel_labels.reshape(1,voxel_labels.shape[0]) #add batch dimension

        data_tuple = (voxel_position,voxel_labels)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz_pol,xyz[:,:2]),axis = 1)

        #point features
        return_fea = np.concatenate((return_xyz,sig[...,np.newaxis]),axis = 1)

        # (grid-wise coor, grid-wise sem label, point-wise grid index, indices voxel to point, point-wise sem label, [relative polar coor(3), polar coor(3), cat coor(2), ref signal(1)])
        data_tuple += (grid_ind, vox_to_point,labels,return_fea)
        offsets = np.zeros([xyz.shape[0], 3], dtype=np.float32)
        offsets = nb_aggregate_pointwise_center_offset(offsets, xyz, ins_labels, self.center_type)

        if len(data) == 9:
            data_tuple += (ins_labels, offsets, valid, xyz, pcd_fname, feats, pos_scans, ids) # plus (point-wise instance label, point-wise center offset)

        elif len(data) == 10:
            data_tuple += (ins_labels, offsets, valid, xyz, pcd_fname, pose, feats, pos_scans, ids) # plus (point-wise instance label, point-wise center offset)

        elif len(data) == 11:
            data_tuple += (ins_labels, offsets, valid, xyz, pcd_fname, feats, pos_scans, ids, pt_sem_pred, pt_ins_pred) # plus (point-wise instance label, point-wise center offset)

        elif len(data) == 12:
            data_tuple += (ins_labels, offsets, valid, xyz, pcd_fname, pose, feats, pos_scans, ids, pt_sem_pred, pt_ins_pred) 

        return data_tuple

def calc_xyz_middle(xyz):
    return np.array([
        (np.max(xyz[:, 0]) + np.min(xyz[:, 0])) / 2.0,
        (np.max(xyz[:, 1]) + np.min(xyz[:, 1])) / 2.0,
        (np.max(xyz[:, 2]) + np.min(xyz[:, 2])) / 2.0
    ], dtype=np.float32)

things_ids = set([10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259])

# @nb.jit
def nb_aggregate_pointwise_center_offset(offsets, xyz, ins_labels, center_type):
    # ins_num = np.max(ins_labels) + 1
    # for i in range(1, ins_num):
    for i in np.unique(ins_labels):
        if (i & 0xFFFF) not in things_ids:
            continue
        i_indices = (ins_labels == i).reshape(-1)
        xyz_i = xyz[i_indices]
        if xyz_i.shape[0] <= 0:
            continue
        if center_type == 'Axis_center':
            mean_xyz = calc_xyz_middle(xyz_i)
        elif center_type == 'Mass_center':
            mean_xyz = np.mean(xyz_i, axis=0)
        offsets[i_indices] = mean_xyz - xyz_i
    return offsets

@nb.jit('u1[:](u1[:],i8[:,:])',nopython=True,cache=True,parallel = False)
def nb_get_voxel_labels(voxel_labels,sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16)
    counter[sorted_label_voxel_pair[0,3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0,:3]
    voxel_ind = 0
    for i in range(1,sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i,:3]
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            voxel_labels[voxel_ind] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
            voxel_ind += 1
        counter[sorted_label_voxel_pair[i,3]] += 1
    voxel_labels[voxel_ind] = np.argmax(counter)
    return voxel_labels

def collate_fn_BEV(data): # stack along batch dimension
    grid_ind_all = [d[2] for d in data]
    label_all = [d[1] for d in data]
    pt_label_all = [d[3] for d in data]
    data2stack=np.stack([d[0] for d in data]).astype(np.float32) # grid-wise coor
    label2stack = [d[1] for d in data]                           # grid-wise sem label
    grid_ind_stack = [d[2] for d in data]                        # point-wise grid index
    vox_to_point = [d[3] for d in data]                          # indices mapping voxel-to-point
    point_label = [d[4] for d in data]                           # point-wise sem label
    xyz = [d[5] for d in data]                                   # point-wise coor

    pt_ins_labels = [d[6] for d in data]                         # point-wise instance label
    pt_offsets = [d[7] for d in data]                            # point-wise center offset
    pt_valid = [d[8] for d in data]                              # point-wise indicator for foreground points
    pt_cart_xyz = [d[9] for d in data]                           # point-wise cart coor
    filename = [d[10] for d in data]                             # scan filename
    pose = [d[11] for d in data]                                 # pose of the scan
    feats = [d[12] for d in data]                                # instance feats (valid)
    pos_scans = [d[13] for d in data]
    ids = [d[14] for d in data]

    out_dict = {
        'vox_coor': torch.from_numpy(data2stack),
        'vox_labels': label2stack,
        'grid': grid_ind_stack,
        'vox2point_idx' : vox_to_point,
        'pt_labs': point_label,
        'pt_fea': xyz,
        'pt_ins_labels': pt_ins_labels,
        'pt_offsets': pt_offsets,
        'pt_valid': pt_valid,
        'pt_cart_xyz': pt_cart_xyz,
        'pcd_fname': filename,
        'pose': pose,
        'feats': feats,                          ##########
        'pos_scans': pos_scans,
        'ids': ids
        }

    if len(data[0]) == 17:
        pt_ins_pred = [d[15] for d in data]
        pt_labs = [d[16] for d in data]
        out_dict['pt_sem_pred'] = pt_ins_pred                #per point sem label
        out_dict['pt_ins_pred'] = pt_labs    #per point instance label

    return out_dict

# Transformations between Cartesian and Polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
    return np.stack((rho,phi,input_xyz[:,2]),axis=1)

def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0]*np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0]*np.sin(input_xyz_polar[1])
    return np.stack((x,y,input_xyz_polar[2]),axis=0)

def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

def absoluteDirPath(directory):
    return os.path.abspath(directory)

def parse_calibration(filename):
    calib = {}
    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        calib[key] = pose
    calib_file.close()
    return calib

def parse_poses(filename, calibration):
    file = open(filename)
    poses = []
    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    for line in file:
        values = [float(v) for v in line.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
    return poses

def load_poses(pose_files, calib_files):
    poses = []
    for i in range(len(pose_files)):
        calib = parse_calibration(calib_files[i])
        seq_poses_f64 = parse_poses(pose_files[i], calib)
        seq_poses = ([pose.astype(np.float32) for pose in seq_poses_f64])
        poses += seq_poses
    return poses
    # poses = []
    # #go through every file and get all poses
    # #add them to match im_idx
    # for i in range(len(pose_files)):
    #     empty = get_empty(empty_files[i])
    #     calib = parse_calibration(calib_files[i])
    #     seq_poses_f64 = parse_poses(pose_files[i], calib)
    #     seq_poses = ([seq_poses_f64[i].astype(np.float32) for i in range(len(seq_poses_f64)) if i not in empty])
    #     poses += seq_poses
    # return poses


def get_empty(filename):
    empty_list = []
    empty_file = open(filename)
    for line in empty_file:
        empty_list.append(int(line.strip()))
    empty_file.close()
    return empty_list

def apply_pose(points, pose):
    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
    shifted_points = torch.tensor(np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)[:,:3])
    return shifted_points