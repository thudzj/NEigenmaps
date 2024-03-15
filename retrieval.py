import argparse
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import numpy as np
import warnings
import os
import timm
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
# import kornia.augmentation as Kg
warnings.filterwarnings("ignore", category=DeprecationWarning)
from collections import OrderedDict

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

from utils import Transform

def load_img(img_name):
    with open(img_name, "rb") as f:
        image = Image.open(f)
        return image.convert("RGB")

class Loader(Dataset):
    def __init__(self, img_dir, txt_dir, NB_CLS=None, transform=None):
        self.img_dir = img_dir
        self.file_list = np.loadtxt(txt_dir, dtype='str')
        self.NB_CLS = NB_CLS
        self.trainsform = transform
        if 'imagenet' in txt_dir:
            self.dset = 'imagenet'
            trainset = torchvision.datasets.ImageFolder(os.path.join(img_dir, 'train'))
            valset = torchvision.datasets.ImageFolder(os.path.join(img_dir, 'val'))
            samples = trainset.samples + valset.samples
            self.headers = list(set(['/'.join(s[0].split('/')[:-1]) for s in samples]))
        elif 'nuswide_m' in txt_dir:
            self.dset = 'nuswide_m'
        else:
            self.dset = 'coco'

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()

        if self.dset == 'imagenet':
            for header in self.headers:
                if os.path.exists(os.path.join(header, self.file_list[idx][0])):
                    break
            img_name = os.path.join(header, self.file_list[idx][0])
        # elif self.dset == 'nuswide_m':
        #     assert False
        else:
            img_name = os.path.join(self.img_dir,
                                    self.file_list[idx][0])
        image = load_img(img_name)

        if self.NB_CLS != None:
            if len(self.file_list[idx])>2:
                label = [int(self.file_list[idx][i]) for i in range(1,self.NB_CLS+1)]
                label = T.FloatTensor(label)
            else:
                label = int(self.file_list[idx][1])
            return self.trainsform(image), label, idx
        else:
            return self.trainsform(image)

@T.no_grad()
def Evaluate_mAP(device, gallery_codes, query_codes, gallery_labels, query_labels, Top_N=None):
    num_query = query_labels.shape[0]
    mean_AP = 0.0
    mean_P = 0.0

    all_retrieval = (query_labels @ gallery_labels.t() > 0).float()

    gallery_codes = F.normalize(gallery_codes, dim=1)
    query_codes = F.normalize(query_codes, dim=1)
    all_hamming_dist = (gallery_codes.shape[1] - query_codes @ gallery_codes.t())
    all_sort_idx = T.cat([T.argsort(all_hamming_dist[i*num_query//10:(i+1)*num_query//10], dim=-1)[..., :Top_N] for i in range(10)], 0)

    retrievals = []
    for i in range(num_query):
        retrieval = all_retrieval[i] #(query_labels[i, :] @ gallery_labels.t() > 0).float()
        # hamming_dist = all_hamming_dist[i] # (gallery_codes.shape[1] - query_codes[i, :] @ gallery_codes.t())

        # sort_idx = T.argsort(hamming_dist.cpu())[:Top_N].to(device)
        sort_idx = all_sort_idx[i] #T.argsort(hamming_dist)[:Top_N]
        retrieval = retrieval[sort_idx]
        retrieval_cnt = retrieval.sum().int().item()

        retrievals.append(sort_idx[:20])

        if retrieval_cnt == 0:
            continue

        score = T.linspace(1, retrieval_cnt, retrieval_cnt).to(device)
        index = (T.nonzero(retrieval == 1, as_tuple=False).squeeze() + 1.0).float().to(device)

        mean_AP += (score / index).mean()
        mean_P += retrieval.mean()

    mean_AP = mean_AP / num_query
    mean_P = mean_P / num_query
    return mean_AP, mean_P, T.stack(retrievals)

def DoRetrieval(device, net, log_dir, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, batch_size, subset=None, dname='coco', random_runs=None, pca=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    Gallery_set = Loader(Img_dir, Gallery_dir, NB_CLS, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    Gallery_loader = T.utils.data.DataLoader(Gallery_set, batch_size=batch_size, num_workers=4, shuffle=True)

    Query_set = Loader(Img_dir, Query_dir, NB_CLS, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    Query_loader = T.utils.data.DataLoader(Query_set, batch_size=batch_size, num_workers=4)

    
    # q_img = Query_set[234][0]
    # q_img = q_img.mul(T.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)).add(T.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
    # print(q_img.min(), q_img.max())
    # q_img = (q_img * 255).clamp(0, 255).to(T.uint8)
    # q_img = q_img.permute(1, 2, 0)
    # q_img = q_img.contiguous().cpu().numpy()
    # fig = plt.figure()
    # fig.set_figheight(4)
    # fig.set_figwidth(44)
    # plt.imsave("{}/test.pdf".format(log_dir), q_img[:,:,:])
    # plt.axis('off')
    # return

    if pca:
        dataset = torchvision.datasets.ImageFolder(os.path.join('/home/data/ImageNet', 'train'), Transform(None))
        loader = T.utils.data.DataLoader(
            dataset, batch_size=1024, num_workers=16,
            pin_memory=True, sampler=None)
        with T.no_grad():
            cov_ = 0
            cnt = 0
            for i, ((data, _), _) in enumerate(loader, 0):
                if i % 100 == 0:
                    print(i)
                data = data.to(device)
                with T.cuda.amp.autocast():
                    outputs = net(data).float()
                cov_ += outputs.T @ outputs
                cnt += outputs.shape[0]
            cov_ /= cnt
            L, Q = T.linalg.eigh(cov_)
            print(L[:5], L[-5:])

    with T.no_grad():
        for i, data in enumerate(Gallery_loader, 0):
            gallery_x_batch, gallery_y_batch, gallery_idx_batch = data[0].to(device), data[1].to(device), data[2].to(device)
            if gallery_y_batch.dim() == 1:
                gallery_y_batch = T.eye(NB_CLS, device=device)[gallery_y_batch]

            outputs = net(gallery_x_batch)

            if i == 0:
                gallery_c = outputs
                gallery_y = gallery_y_batch
                gallery_idx = gallery_idx_batch
            else:
                gallery_c = T.cat([gallery_c, outputs], 0)
                gallery_y = T.cat([gallery_y, gallery_y_batch], 0)
                gallery_idx = T.cat([gallery_idx, gallery_idx_batch], 0)

            # if subset is not None and i == 50:
            #     break

        for i, data in enumerate(Query_loader, 0):
            query_x_batch, query_y_batch, query_idx_batch = data[0].to(device), data[1].to(device), data[2].to(device)
            if query_y_batch.dim() == 1:
                query_y_batch = T.eye(NB_CLS, device=device)[query_y_batch]

            outputs = net(query_x_batch)

            if i == 0:
                query_c = outputs
                query_y = query_y_batch
                query_idx = query_idx_batch
            else:
                query_c = T.cat([query_c, outputs], 0)
                query_y = T.cat([query_y, query_y_batch], 0)
                query_idx = T.cat([query_idx, query_idx_batch], 0)

            if subset is not None and i == subset:
                break

    # gallery_c = T.sign(gallery_c)
    # query_c = T.sign(query_c)

    ks = [4]
    while ks[-1] * 2 <= query_c.shape[-1]:
        ks.append(ks[-1] * 2)
    if ks[-1] != query_c.shape[-1]:
        ks.append(query_c.shape[-1])

    # print(ks)
    maps = []
    ppp = T.randperm(query_c.shape[0])[:10]
    if random_runs:
        rand_indices = T.argsort(T.rand(random_runs, gallery_c.shape[-1]), dim=-1)

    for k in ks:
        if random_runs:
            mAP, mean_P = [], []
            for run in range(random_runs):
                gallery_c_k = gallery_c[...,rand_indices[run, :k]]
                query_c_k = query_c[...,rand_indices[run, :k]]

                mAP_, mean_P_, retrievals = Evaluate_mAP(device, gallery_c_k, query_c_k, gallery_y, query_y, Top_N)
                mAP.append(mAP_.cpu()); mean_P.append(mean_P_.cpu())
            print('     k: %.0f  mAP: %.4f %.4f  mean_P: %.4f %.4f' %
                (k,np.mean(mAP),np.std(mAP),np.mean(mean_P),np.std(mean_P)))
            maps.append(np.mean(mAP))
        else:
            if pca:
                gallery_c_k = gallery_c @ Q[:, -k:]
                query_c_k = query_c @ Q[:, -k:]
            else:
                gallery_c_k = gallery_c[...,:k]
                query_c_k = query_c[...,:k]

            mAP, mean_P, retrievals = Evaluate_mAP(device, gallery_c_k, query_c_k, gallery_y, query_y, Top_N)
            maps.append(mAP)
            print('     k: %.0f  mAP: %.4f  mean_P: %.4f' % (k, mAP, mean_P))

        for ite, i in enumerate(ppp):

            q_img = Query_set[query_idx[i]][0]
            q_img = (q_img.mul(T.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)).add(T.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) * 255).clamp(0, 255).to(T.uint8)
            q_img = q_img.permute(1, 2, 0)
            q_img = q_img.contiguous().cpu().numpy()

            v_imgs = [Gallery_set[gallery_idx[j]][0] for j in retrievals[i]]
            v_imgs = T.stack(v_imgs)[:10]
            num_v_imgs = v_imgs.shape[0]
            v_imgs = (v_imgs.mul(T.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)).add(T.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)) * 255).clamp(0, 255).to(T.uint8)
            v_imgs = v_imgs.permute(0, 2, 3, 1)
            v_imgs = v_imgs.contiguous().cpu().numpy()
            v_imgs = v_imgs.reshape((1, num_v_imgs, 224, 224, 3)).transpose((0, 2, 1, 3, 4)).reshape((1 * 224, num_v_imgs * 224, 3))

            img = np.concatenate([q_img, np.zeros((224, 10, 3)).astype(np.uint8), v_imgs], 1)

            im = Image.fromarray(img).convert('RGB')
            im.save("{}/{}_{}_{}.pdf".format(log_dir, dname, k, ite))

            # plt.figure(figsize=(4, 44))
            # plt.imsave("{}/retrieval_{}_{}.pdf".format(log_dir, k, ite), img[:,:,:])
            # plt.axis('off')
    return maps

    # for k, mAP in zip(ks, maps):


def retrieval(Img_dir, Gallery_dir, Query_dir, device, model_fn, batch_size, log_dir, subset=None, dname='coco', random_runs=None, pca=False):
    if dname=='coco':
        NB_CLS=80
        Top_N=5000
    elif dname=='imagenet':
        NB_CLS=100
        Top_N=100
    elif dname=='nuswide':
        NB_CLS=21
        Top_N=5000
    elif dname=='nuswide_m':
        NB_CLS=21
        Top_N=5000
    elif dname=='voc2012':
        NB_CLS=20
        Top_N=100
    elif dname=='mirflickr':
        NB_CLS=38
        Top_N=100
    else:
        print("Wrong dataset name.")
        return
    return DoRetrieval(device, model_fn, log_dir, Img_dir, Gallery_dir, Query_dir, NB_CLS, Top_N, batch_size, subset, dname, random_runs, pca)
