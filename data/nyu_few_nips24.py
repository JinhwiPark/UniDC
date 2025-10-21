import os
import warnings
import numpy as np
import json
import h5py

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

warnings.filterwarnings("ignore", category=UserWarning)

MEAN=(123.675, 116.28, 103.53)
STD=(58.395, 57.12, 57.375)


class BaseDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    # A workaround for a pytorch bug
    # https://github.com/pytorch/vision/issues/2194
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)

class NYU(BaseDataset):
    def __init__(self, args, mode, num_sample_test=None):
        super(NYU, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.num_sample = num_sample_test
        if args.model_name == 'LRRU':
            from data.lrru.NNfill import fill_in_fast
            self.ipfill = fill_in_fast

        if self.mode =='test':
            assert type(self.num_sample) is int, "TEST dataset should have specific # of sample !!"

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        self.height = args.patch_height
        self.width = args.patch_width
        
        # print(args)
        # print(self.height)
        if self.height==240:
            self.crop_size = (228,304)
        elif self.height==480:
            self.crop_size = (456,608)
        elif self.height==256:
            self.crop_size = (228,304)
        elif 'midas' in args.model_name:
            self.crop_size = (240,320)
        else:
            raise print("Check the self.height !!")
        print("Crop Size:",self.crop_size)

        # Camera intrinsics [fx, fy, cx, cy]
        self.K = torch.Tensor([
            5.1885790117450188e+02 / 2.0,
            5.1946961112127485e+02 / 2.0,
            3.2558244941119034e+02 / 2.0 - 8.0,
            2.5373616633400465e+02 / 2.0 - 6.0
        ])
        self.augment = False if self.mode == 'test' else self.args.augment
        # print("Augmentation: ",self.augment)
        # self.augment = False 

        # import pdb;pdb.set_trace()
        
        with open(self.args.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

            # random 1/10/100/1000-shot
            if args.minidataset_fewshot:
                print("Few-shot for NYU minidataset !!")
                if args.few_shot_way == 'random' and self.mode =='train':
                    print(" [Random] !! ==>", args.minidataset_fewshot_number)
                    import random
                    random.seed(args.seed)
                    self.sample_list_new = random.sample(self.sample_list,args.minidataset_fewshot_number)
                    self.sample_list = self.sample_list_new
                elif args.few_shot_way == 'one_sequence' and self.mode =='train':
                    print(" [ONE SEQUENCE] !! ==>", args.nyu_sequence_name)
                    self.sample_list_new=[]
                    for i in self.sample_list:
                        if args.nyu_sequence_name in i['filename']:
                            self.sample_list_new.append(i)
                    self.sample_list = self.sample_list_new
                    print("# of dataset :", len(self.sample_list))
                elif args.few_shot_way == '1-shot_for_each_sequence' and self.mode =='train':
                    print(" [1-SHOT FOR EACH SEQUENCE] !!")
                    with open('./data/nyu_one-shot_for_each_sequence.json') as json_file:
                        json_data = json.load(json_file)
                        self.sample_list = json_data[mode]
                        print("# of dataset :", len(self.sample_list))
                elif args.few_shot_way == '1-percent_for_each_sequence' and self.mode =='train':
                    print(" [1+% FOR EACH SEQUENCE] !!")
                    with open('./data/nyu_1percent_for_each_sequence.json') as json_file:
                        json_data = json.load(json_file)
                        self.sample_list = json_data[mode]
                        print("# of dataset :", len(self.sample_list))


            else: print("This dataloader for Few-shot Learning"); NotImplementedError
        
    

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()
        path_file = os.path.join(self.args.dir_data,
                                 self.sample_list[idx]['filename'])

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
        dep_h5 = f['depth'][:]

        # print(rgb_h5.shape) -> (480,640,3)
        # print(dep_h5.shape) -> (480,640)
        rgb = Image.fromarray(rgb_h5, mode='RGB')
        dep = Image.fromarray(dep_h5.astype('float32'), mode='F')
        
        rgb_480640 = 0
        dep_480640 = 0
        # if self.args._480640:
        #     t_rgb_480640 = T.Compose([
        #                             T.Resize(480),
        #                             T.CenterCrop((480,640)),
        #                             T.ToTensor(),
        #                             T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #                             ])

        #     rgb_480640 = t_rgb_480640(rgb)
        
        if self.mode == 'train' and self.augment:
            
            # Augmentation 적용하는 것이 더 안좋은듯
            # Pretrained Weight를 그대로 쓸려면 그냥 넣어주는게 좋다. -> FIXME: Bug..

            _scale = np.random.uniform(1.0, 1.5)
            scale = int(self.height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            if flip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)

            rgb = TF.rotate(rgb, angle=degree)#, resample=Image.NEAREST)
            dep = TF.rotate(dep, angle=degree)#, resample=Image.NEAREST)


                
            t_rgb = T.Compose([
                T.Resize(scale),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_dep = T.Compose([
                T.Resize(scale),
                T.CenterCrop(self.crop_size),
                self.ToNumpy(),
                T.ToTensor()
            ])

            if self.args._480640:
                rgb_480640=rgb
                t_rgb_480640 = T.Compose([
                                        T.Resize(scale),
                                        T.CenterCrop(self.crop_size),
                                        T.Resize((456,608)),
                                        T.ToTensor(),
                                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])
                rgb_480640 = t_rgb_480640(rgb_480640)

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            dep = dep / _scale

            K = self.K.clone()
            K[0] = K[0] * _scale
            K[1] = K[1] * _scale
            

        else:
            t_rgb = T.Compose([
                T.Resize(self.height), # 480 640 -> 240 320
                T.CenterCrop(self.crop_size), # 240 320 -> 228 304
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                T.Resize(self.height), # 480 640 -> 240 320
                T.CenterCrop(self.crop_size), # 240 320 -> 228 304
                self.ToNumpy(),
                T.ToTensor()
            ])

            if self.args._480640:
                rgb_480640=rgb
                t_rgb_480640 = T.Compose([
                                        T.Resize(480),
                                        T.CenterCrop((456,608)),
                                        T.ToTensor(),
                                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])
                rgb_480640 = t_rgb_480640(rgb_480640)
                
                dep_480640=dep
                t_dep_480640 = T.Compose([
                                        T.Resize(480),
                                        T.CenterCrop((456,608)),
                                        self.ToNumpy(),
                                        T.ToTensor()
                                        ])
                dep_480640 = t_dep_480640(dep_480640)
                
                
            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            K = self.K.clone()

        if self.args.affinity_equivariance:
            if self.mode =='test':
                dep_sp_1,ns = self.get_sparse_depth(dep, self.num_sample, test=True)
                output = {'rgb': rgb, 'dep_sp_1': dep_sp_1, 'dep_sp_2': dep_sp_1, 'gt': dep, 'K': K, 'rgb_480640':rgb_480640, 'dep_480640':dep_480640, 'num_sample':ns}
            else:
                dep_sp_1, ns = self.get_sparse_depth(dep, self.args.num_sample, test=False, max_=self.args.sp_max)
                dep_sp_2, _ = self.get_sparse_depth(dep, self.args.num_sample, test=False, max_=self.args.sp_max)
                output = {'rgb': rgb, 'dep_sp_1': dep_sp_1, 'dep_sp_2': dep_sp_2, 'gt': dep, 'K': K, 'rgb_480640':rgb_480640, 'dep_480640':dep_480640, 'num_sample':ns}
            return output
        else:
            dep_sp,ns = self.get_sparse_depth(dep, self.args.num_sample, test=False, max_=self.args.sp_max)
        
        if self.mode =='test':
            # dep_sp = self.get_sparse_depth(dep, self.args.num_sample, test=True)
            dep_sp,ns = self.get_sparse_depth(dep, self.num_sample, test=True)


        if self.args.model_name == 'BPNet_nopad':
            K=torch.tensor([[259.4290,   0.0000, 162.7912],
            [  0.0000, 259.7348, 134.8681],
            [  0.0000,   0.0000,   1.0000]])

        # elif self.args.model_name == 'BPNet' or self.args.model_name == 'BPNet_nomul' or self.args.model_name == 'BPNet_foundation':
        elif 'BPNet' in self.args.model_name:
            rgb = TF.pad(rgb, padding=[8, 14], padding_mode='edge')
            dep_sp = TF.pad(dep_sp, padding=[8, 14], padding_mode='constant')
            dep = TF.pad(dep, padding=[8, 14], padding_mode='constant')
            K=torch.tensor([[259.4290,   0.0000, 162.7912],
            [  0.0000, 259.7348, 134.8681],
            [  0.0000,   0.0000,   1.0000]])
            

        dep_ip_torch=0
        if self.args.model_name == 'LRRU':
            dep_np_ip = np.copy(dep_sp.numpy())
            dep_ip = self.ipfill(dep_np_ip, max_depth=10.0, extrapolate=False, blur_type='gaussian')
            dep_ip_torch = torch.from_numpy(dep_ip)
            dep_ip_torch = dep_ip_torch.to(dtype=torch.float32)

        # else: dep_sp,ns = self.get_sparse_depth(dep, self.args.num_sample, test=False, max_=self.args.sp_max)

        output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K, 'rgb_480640':rgb_480640, 'dep_480640':dep_480640, 'num_sample':ns, 'prefill_depth':dep_ip_torch}
        return output

    def get_sparse_depth(self, dep, num_sample, test=False, max_=500):
        channel, height, width = dep.shape

        assert channel == 1

        if self.args.exp_name == 'TestSparse':
            num_sample=50
        if self.args.exp_name == 'TestGrid': # PatternChange 실험 -> Grid로 줘야함

            h=torch.arange(5,228,50)
            w=torch.arange(5,304,50)
            mask_h = torch.zeros(dep.shape)
            mask_w = torch.zeros(dep.shape)
            mask_h[:,h,:]=1.
            mask_w[:,:,w]=1.
            dep = dep * mask_h * mask_w
            return dep, dep.nonzero().shape[0]
        elif self.args.exp_name == 'TestGrid2': # PatternChange 실험 -> Grid로 줘야함

            h=torch.arange(5,228,30)
            w=torch.arange(5,304,30)
            mask_h = torch.zeros(dep.shape)
            mask_w = torch.zeros(dep.shape)
            mask_h[:,h,:]=1.
            mask_w[:,:,w]=1.
            dep = dep * mask_h * mask_w
            return dep, dep.nonzero().shape[0]

        if self.args.exp_name == 'TestFar': # 가까운 곳만 train -> 확인은 먼 곳만
            idx_nnz = torch.nonzero(dep.view(-1)> 3.0, as_tuple=False)
            num_sample=200
        elif self.args.exp_name == 'TestNear': # 먼 곳만 train -> 확인은 가까운 곳만
            mask = torch.logical_and(dep > 1e-3, dep < 3.0)
            idx_nnz = torch.nonzero(mask.view(-1), as_tuple=False)
            num_sample=200
        else:
            idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)

        if test:
            g_cpu = torch.Generator()
            # g_cpu.manual_seed(77)
            g_cpu.manual_seed(self.args.sample_seed)
            idx_sample = torch.randperm(num_idx, generator=g_cpu)[:num_sample]
        else:
            if num_sample == 'random':
                num_sample = random.randint(1, max_)
            else:
                num_sample = int(num_sample)
            idx_sample = torch.randperm(num_idx)[:num_sample]           
        

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)
        # print(dep_sp.nonzero().shape)
        return dep_sp, num_sample
    