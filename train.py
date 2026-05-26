import torch.nn as nn
from ResNetSA import get_model
from GCN.GCN import TwoLayerGCN,build_topk_adj,build_topk_adj_gat,GAN
from torch.utils.data import Dataset, DataLoader,ConcatDataset
import itertools
import importlib
from utils import *
from loss import supcon_loss
from loss.lossjs import consistency_loss,graph_consistency_loss,GraphSpectralConsistencyLoss
from itertools import chain

import time
import numpy as np
from torchvision import transforms, datasets
import argparse

from datasets.supcon_dataset import FaceDataset, DEVICE_INFOS,JPEGCompression,AddCameraSensorNoise

from datasets import get_datasets, TwoCropTransform,get_single_dataset
from datasets.data.base import data_aug
from datasets.data.transform import get_basetransform,random_parse_policies, MultiAugmentation
from datasets.data.FAS_Augmentations import LowResolutionAug,HandTremblingAug,ColorDiversityAug
from thop import profile
from fvcore.nn import FlopCountAnalysis

from torchinfo import summary

torch.backends.cudnn.benchmark = True

class GCNWrapper(torch.nn.Module):
    def __init__(self, gcn, A_hat):
        super().__init__()
        self.gcn = gcn
        self.register_buffer("A_hat", A_hat)

    def forward(self, x):
        return self.gcn(x, self.A_hat)
    
def log_f(f, console=True):
    def log(msg):
        with open(f, 'a') as file:
            file.write(msg)
            file.write('\n')
        if console:
            print(msg)
    return log

def binary_func_sep(logits, label, ce_loss_record):
    """
    logits: [B, 2]
    label : [B] (0 or 1)
    """

    criterion = nn.CrossEntropyLoss().cuda()

    # --- 分类损失 ---
    label = label.long().cuda()        # CE 要 long
    logits = logits.cuda()

    cls_loss = criterion(logits, label)

    # --- 预测 ---
    pred = torch.argmax(logits, dim=1)  # [B]

    # --- accuracy ---
    correct = (pred == label).sum().item()
    total = label.size(0)

    # --- 记录 loss ---
    ce_loss_record.update(cls_loss.item(), total)

    return cls_loss, (correct, total)

def main(args):
    
    data_name_list_train, data_name_list_test = protocol_decoder(args.protocol)
    if args.pretrain == 'imagenet':
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_transform_list = [
        transforms.RandomResizedCrop(256, scale=(args.train_scale_min, 1.), ratio=(1., 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer
    ]

    if args.train_rotation:
        train_transform_list = [transforms.RandomRotation(degrees=(-180, 180))] + train_transform_list
        
    transform_extra = [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply([HandTremblingAug(k_min=2, k_max=8)], p=0.15),
        #transforms.RandomApply([LowResolutionAug(sr_min=0.2, sr_max=0.5)], p=0.1),
        #transforms.RandomApply([ColorDiversityAug(rgb_profile_path='datasets/data/profile/RGB Profiles/')], p=0.1),
        #transforms.ToTensor(),
    ]
    if args.train_extra:
        train_transform_list = transform_extra + train_transform_list
   
    train_transform = transforms.Compose(train_transform_list)
   
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalizer
    ])
    
    dataset_base = get_datasets(
        args.data_dir, FaceDataset,
        train=True,
        protocol=args.protocol,  # 或 data_name_list_train[i % len(data_name_list_train)]
        img_size=args.img_size,
        map_size=32,
        transform=train_transform,
        debug_subset_size=args.debug_subset_size
    )
    train_loader = DataLoader(dataset_base, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    test_set = get_datasets(args.data_dir, FaceDataset, train=False, protocol=args.protocol, img_size=args.img_size, map_size=32, transform=test_transform, debug_subset_size=args.debug_subset_size)
    test_loader = DataLoader(test_set[data_name_list_test[0]], batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    live_cls_list = []
    spoof_cls_list = []
    for dataset in data_name_list_train:
        live_cls_list += DEVICE_INFOS[dataset]['live']
        spoof_cls_list += DEVICE_INFOS[dataset]['spoof']
    total_cls_num = 2

    device2idx = {pattern: idx for idx, pattern in enumerate(spoof_cls_list)}

    max_iter = args.num_epochs*len(train_loader)
    # make dirs
    model_root_path = os.path.join(args.result_path, args.result_name, "model")
    check_folder(model_root_path)
    score_root_path = os.path.join(args.result_path, args.result_name, "score")
    check_folder(score_root_path)
    log_path = os.path.join(args.result_path, args.result_name, "log.txt")
    print = log_f(log_path)

    if args.pretrain == 'imagenet':
        model = get_model(args)   # Feature Net
        # 初始化 GCN
        if args.model_name1  == 'gcn':
            gcn = TwoLayerGCN(
                in_dim=512,   #resnet=512  Mobilenet_large = 960
                hidden_dim=256,
                num_classes=2,
                dropout = 0.2
            ).cuda()
        elif args.model_name1  == 'gat':
            feat1_dim = 512
            gan = GAN(
                n_feat=feat1_dim,
                n_hidden=64,
                n_class=2,
                dropout=0.6,
                alpha=0.2,
                n_heads=8
            ).to(device)     

    else:
        model = get_model(args)
        model_path = os.path.join('pretrained', args.pretrain, 'model', "{}_best.pth".format(args.pretrain))
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])

    model = model.cuda()
   
    optimizer1 = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # def scheduler
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    
    optimizer2 = torch.optim.SGD(gcn.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer2 = torch.optim.SGD(gan.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # def scheduler
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=args.step_size, gamma=args.gamma)

    if args.resume:
        model_path = os.path.join(model_root_path, "{}_p{}_best.pth".format(args.model_type, args.protocol))
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        # args.start_epoch = ckpt['epoch']
        scheduler = ckpt['scheduler']
    
    # metrics
    eva = {
        "best_epoch": -1,
        "best_HTER": 100,
        "best_auc": -100
    }

    #ce_loss = nn.BCELoss().cuda()
    epoch_times = []
    #latent_list = []
    #label_list = []
    #global_latent = torch.empty(0)
    #global_label = torch.empty(0)
    for epoch in range(args.start_epoch, args.num_epochs):
        #ce_loss_record_0 = AvgrageMeter()
        #ce_loss_record_1 = AvgrageMeter()
        #ce_loss_record_2 = AvgrageMeter()
        ce_loss_record = AvgrageMeter()
        feat_loss_record = AvgrageMeter()
        #graph_loss_record = AvgrageMeter()
        patch_loss_record = AvgrageMeter()
        loss_all_record = AvgrageMeter()
        ########################### train ###########################
        model.train()
        gcn.train()
        #gan.train()
        #correct = 0
        #total = 0
        # 动态调整 k
        #k = min(30, 2 + epoch // 5)
        for j, sample_batched in enumerate(train_loader):
            lr = optimizer1.param_groups[0]['lr']

            # --- 数据加载 ---
            image_x_o = sample_batched["image_x_o"].cuda() #原始图像
            image_x_v1 = sample_batched["image_x_v1"].cuda()  #增强图像1
            image_x_v2 = sample_batched["image_x_v2"].cuda()  #增强图像2
            label = sample_batched["label"].cuda()    #图像标签
            UUID = sample_batched["UUID"].cuda()      #图像的域ID = （0,1,2）
            
            # 统计各类别数量（假设1=live, 0=spoof）  
            live_n = torch.sum(label == 1).item()  
            spoof_n = torch.sum(label == 0).item() 

            image_x = torch.cat([image_x_v1, image_x_v2])
            UUID2 = torch.cat([UUID, UUID])
            label2 = torch.cat([label, label])
            
            feat_o,_ = model(image_x_o, out_type='feat')   #batch 原样本特征
            feat1,patch_feat1  = model(image_x_v1, out_type='feat')   #batch 原样本增强view1特征
            feat2,patch_feat2 = model(image_x_v2, out_type='feat')   #batch 原样本增强view2特征
            feat,patch_feat = model(image_x, out_type='feat')   #batch 原样本增强（两个view）特征

            
            feat_o = F.normalize(feat_o)  #batch 原样本特征
            feat1 = F.normalize(feat1)  #batch 增强样本view1特征
            feat2 = F.normalize(feat2)  #batch 增强样本view2特征
            feat = F.normalize(feat)
            #patch_feat  = F.normalize(patch_feat)
            #patch_feat = patch_feat.detach()
            # 前向传播
            #logits = model(image_x_v1, out_type='ce') 
            top_k = min(live_n, spoof_n,args.top_k)
            A_hat = build_topk_adj(feat,top_k)
            _,logits = gcn(feat, A_hat)  # [B, 2]
            
            #adj = build_topk_adj_gat(feat1,top_k)
            #logits = gan(feat1, adj)     # [B, 2]
            
            #f1, f2 = torch.split(feat_normed, [len(image_x_v1), len(image_x_v1)], dim=0)
            
            #cls_loss, stat = binary_func_sep(logits, label, ce_loss_record)
            #correct,total = stat
            #'''
            if args.feat_loss == 'supcon':
                feat_loss = supcon_loss(torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1), UUID * 10 + label, temperature=args.temperature)
            else:
                feat_loss = torch.zeros(1).cuda()    
            #'''
            # 节点级分类
            #loss_all = F.cross_entropy(logits, label2)  # label ∈ [B]    
            #in_channels = patch_feat.shape[1]
            #'''
            patch_loss_fn = GraphSpectralConsistencyLoss(in_channels=256, proj_dim=64).cuda()
            patch_loss = patch_loss_fn(patch_feat1)
            #'''
            #################################################################
            
            if epoch > args.warmup_epochs:
                '''
                graph_loss = graph_consistency_loss(
                    feat_o,
                    f1,
                    global_latent.to(feat_o.device),
                    k=5,
                    top_M=10
                )
                '''
                cls_loss, stat = binary_func_sep(logits, label, ce_loss_record)
                correct,total = stat
                loss_all = cls_loss + args.feat_loss_weight* feat_loss + args.graph_loss_weight*patch_loss
            else:
                #cls_loss, stat = binary_func_sep(logits, label2, ce_loss_record)
                #correct,total = stat
                #loss_all = cls_loss + args.feat_loss_weight * feat_loss + args.graph_loss_weight*patch_loss
                loss_all = args.feat_loss_weight * feat_loss + args.graph_loss_weight*patch_loss
                correct=0
                total=args.batch_size*2
            
            #else:
            #loss_all = cls_loss + args.feat_loss_weight * feat_loss + args.graph_loss_weight*patch_loss 
            
            #loss_all = cls_loss 
            '''
                if epoch == args.warmup_epochs:
                    latent_list.append(feat_o.detach().cuda())  
                    label_list.append(label.detach().cuda())
            '''  
            model.zero_grad()
            gcn.zero_grad()
            #gan.zero_grad()
            loss_all.backward()
            optimizer1.step()
            optimizer2.step()
            feat_loss_record.update(feat_loss.data.item(), len(image_x))
            patch_loss_record.update(patch_loss.data.item(), len(image_x))
            loss_all_record.update(loss_all.data.item(), len(image_x))
            '''
            if epoch > args.warmup_epochs:
                graph_loss_record.update(graph_loss.data.item(), len(image_x_v1))
            
            # 根据配置调整分类对齐权重
            if (epoch+1) >= args.align_epoch and args.align == 'v4'and (epoch+1) % 5== 0:#and epoch % 2 == 0 
                angle = model.update_weight_v4(epoch, args.align_decay_rate)
            else:
                angle = -1.0
            '''
            #log_info = "epoch:{:d}, batch:{:d}, lr={:.4f}, loss_all={:.4f}".format(epoch + 1, j+1, lr ,loss_all_record.avg)
            
            log_info = "epoch:{:d}, mini-batch:{:d}, lr={:.4f}, feat_loss={:.4f},patch_loss={:.4f}, ce_loss_avg={:.4f}, ACC_avg={:.4f}".format(epoch + 1, j+1, lr, feat_loss_record.avg, patch_loss_record.avg,ce_loss_record.avg, 100. * correct/total)
            
            if j % args.print_freq == args.print_freq - 1:
                print(log_info)
        '''
        if epoch == args.warmup_epochs:
            if len(latent_list) > 0:
                global_latent = torch.cat(latent_list, dim=0).cuda()  # concat 后放 GPU
                global_label  = torch.cat(label_list, dim=0)    # [N]
            else:
                global_latent = torch.empty((0, feat_o.shape[1]), device=feat_o.device)
        '''
        # whole epoch average
        print("epoch:{:d}, Train: lr={:4f}, Loss={:.4f}".format(epoch + 1, lr, loss_all_record.avg))
        scheduler1.step()
        scheduler2.step()

        ############################ test ###########################
        if args.protocol  == "I_C_M_to_O":
            epoch_test = 1
        else:
            epoch_test = args.eval_preq

        if epoch > args.warmup_epochs and (epoch % epoch_test == epoch_test-1):
            check_folder(score_root_path)

            model.eval()
            gcn.eval()
            #gan.eval()
            with torch.no_grad():
                start_time = time.time()
                scores_list = []
                for i, sample_batched in enumerate(test_loader):
                    image_x, live_label, UUID = sample_batched["image_x_v1"].cuda(), sample_batched["label"].cuda(), sample_batched["UUID"].cuda()
                    #'''
                    penul_feat, _ = model(image_x, out_type='feat')
                    feat_n = F.normalize(penul_feat)  #batch 原样本特征
                    A_hat = build_topk_adj(feat_n, args.top_k)
                    _,logits = gcn(feat_n, A_hat)  # [B, 2]
                    #'''
                    '''
                    penul_feat, _ = model(image_x, out_type='feat')
                    feat_n = F.normalize(penul_feat)  #batch 原样本特征
                    adj = build_topk_adj_gat(feat_n, args.top_k)
                    logits = gan(feat_n, adj)     # [B, 2]
                    '''
                    #logits = model(image_x, out_type='ce')
                    #########
                    probs = torch.softmax(logits, dim=1)
                    scores = probs[:, 1]              # live score

                    for j in range(scores.size(0)):
                        scores_list.append(
                            "{} {}\n".format(scores[j].item(), live_label[j].item())
                        )
            
            map_score_val_filename = os.path.join(score_root_path, "{}_epoch_{}_score.txt".format(data_name_list_test[0], epoch+1))
            print("score: write test scores to {}".format(map_score_val_filename))
            with open(map_score_val_filename, 'w') as file:
                file.writelines(scores_list)

            test_ACC, fpr, FRR, HTER, auc_test, test_err, tpr = performances_val(map_score_val_filename)
            print("epoch:{:d}, test:  val_ACC={:.4f}, HTER={:.4f}, AUC={:.4f},TPR={:.4f}, val_err={:.4f}, ACC={:.4f} ".format(epoch + 1, test_ACC, HTER, auc_test,tpr, test_err, test_ACC ))
            end_time = time.time()
            test_time = end_time - start_time
            epoch_times.append(test_time)
            print("test phase cost {:.4f}s".format(test_time))
            ############保存最佳模型######################
            if auc_test-HTER>=eva["best_auc"]-eva["best_HTER"]:
                eva["best_auc"] = auc_test
                eva["best_HTER"] = HTER
                eva["tpr95"] = tpr
                eva["best_epoch"] = epoch+1
                model_path = os.path.join(model_root_path, "{}_p{}_best.pth".format(args.model_type, args.protocol))
                '''
                torch.save({
                    'epoch': epoch+1,
                    'state_dict':model.state_dict(),
                    'optimizer':optimizer1.state_dict(),
                    'scheduler':scheduler1,
                    'args':args,
                    'eva': (HTER, auc_test)
                }, model_path)
                '''
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),     # 特征提取器
                    'gcn_state_dict': gcn.state_dict(),         # GCN
                    'optimizer1': optimizer1.state_dict(),
                    'optimizer2': optimizer2.state_dict(),  # 如果 GCN 有单独的 optimizer
                    'scheduler1': scheduler1.state_dict(),         # 学习率调度器
                    'scheduler2': scheduler2.state_dict(),         # 学习率调度器
                    'args':args,
                    'eva': (HTER, auc_test)
                }, model_path)
                print("Model saved to {}".format(model_path))

            print("[Best result] epoch:{}, HTER={:.4f}, AUC={:.4f}".format(eva["best_epoch"],  eva["best_HTER"], eva["best_auc"]))

            model_path = os.path.join(model_root_path, "{}_p{}_recent.pth".format(args.model_type, args.protocol))
            ############保存最近一次模型######################
            '''
            torch.save({
                'epoch': epoch+1,
                'state_dict':model.state_dict(),
                'optimizer':optimizer1.state_dict(),
                'scheduler':scheduler1,
                'args':args,
                'eva': (HTER, auc_test)
            }, model_path)
            '''
            ####################
            torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),     # 特征提取器
                    'gcn_state_dict': gcn.state_dict(),         # GCN
                    'optimizer1': optimizer1.state_dict(),
                    'optimizer2': optimizer2.state_dict(),  # 如果 GCN 有单独的 optimizer
                    'scheduler1': scheduler1.state_dict(),         # 学习率调度器
                    'scheduler2': scheduler2.state_dict(),         # 学习率调度器
                    'args':args,
                    'eva': (HTER, auc_test)
            }, model_path)
    
    # 获取包含 epoch 信息的文件
    saved_files = [file for file in os.listdir(score_root_path) if "_epoch_" in file and file.endswith("_score.txt")]
    # 提取 epoch 信息并转换为整数
    epochs_saved = [(int(file.split("_epoch_")[1].split("_")[0]), file) for file in saved_files]
    # 按 epoch 进行排序（升序）
    epochs_saved = sorted(epochs_saved, key=lambda x: x[0])
    # 获取最近的 10 个 epoch 文件（降序）
    last_n_epochs = epochs_saved[::-1][:20]
    #print(last_n_epochs)

    HTERs, AUROCs, TPRs = [], [], []
    for epoch, file_name in last_n_epochs:
        file_path = os.path.join(score_root_path, file_name)
        test_ACC, fpr, FRR, HTER, auc_test, test_err, tpr = performances_val(file_path)
        HTERs.append(HTER)
        AUROCs.append(auc_test)
        TPRs.append(tpr)
        #print("## {} score:".format(data_name_list_test[0]))
        print("epoch:{:d}, test:  val_ACC={:.4f}, HTER={:.4f}, AUC={:.4f}, TPR={:.4f}, val_err={:.4f}, ACC={:.4f}".format(epoch, test_ACC, HTER, auc_test, tpr, test_err, test_ACC))

    os.makedirs('summary', exist_ok=True)
    file = open(f"summary\\{args.result_name}.txt", "a")
    L = [f"{args.summary}\t\t{eva['best_epoch']}\t{eva['best_HTER']*100:.2f}\t{eva['best_auc']*100:.2f}" +
f"\t{np.array(HTERs).mean()*100:.2f}\t{np.array(HTERs).std()*100:.2f}\t{np.array(AUROCs).mean()*100:.2f}\t{np.array(AUROCs).std()*100:.2f}\t"+f"{np.array(TPRs).mean()*100:.2f}\t{np.array(TPRs).std()*100:.2f}\n"]
    
    # 统计总参数和可训练参数
    total_params1 = sum(p.numel() for p in model.parameters())
    trainable_params1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params2= sum(p.numel() for p in gcn.parameters())
    trainable_params2 = sum(p.numel() for p in gcn.parameters() if p.requires_grad)
    total_params = total_params1 + total_params2
    trainable_params =trainable_params1 + trainable_params2

    print(f"总参数数量 (Total Parameters): {total_params / 1e6:.2f} M")
    print(f"可训练参数数量 (Trainable Parameters): {trainable_params / 1e6:.2f} M")
    #'''
    # FLOPs 计算
    # 输入张量
    input_tensor = torch.randn(1, 3, 256, 256).cuda()
    input_tensor1 = torch.randn(80, 3, 256, 256).cuda()
    flops1 = FlopCountAnalysis(model, input_tensor).total()
    # GCN FLOPs（20 个节点） 
    node_feat = torch.randn(30, 512).cuda() 
    A_hat = torch.randn(30, 30).cuda() 
    gcn_wrapper = GCNWrapper(gcn, A_hat) 
    flops2 = FlopCountAnalysis(gcn_wrapper, node_feat).total()
    flops = flops1 + flops2
    print(f"FLOPs: {flops / 1e9:.2f} G")
    #summary(model, input_size=(1, 3, 256, 256))
    #summary(model, input_size=(1, 3, 256, 256))
    #'''
    # 计算平均epoch时间
    avg_time = sum(epoch_times) / len(epoch_times)
    print(f"Average epoch time: {avg_time:.4f} seconds")
    
    file.writelines(L)
    file.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="datasets/FAS", help='YOUR_Data_Dir')
    parser.add_argument('--result_path', type=str, default='./results', help='root result directory')
    
    #parser.add_argument('--protocol', type=str, default="O_C_I_to_M",
    #                    help='O_C_I_to_M, O_M_I_to_C, O_C_M_to_I, I_C_M_to_O')
    #parser.add_argument('--protocol', type=str, default="M_I_to_C",
    #                    help='O_M_I_to_C, O_M_to_C, O_I_to_C,M_I_to_C,I_to_C')
    #parser.add_argument('--protocol', type=str, default="O_M_to_I",
    #                    help='O_C_M_to_I,O_C_to_I, O_M_to_I,C_M_to_I,C_to_I')
    #parser.add_argument('--protocol', type=str, default="I_C_to_O",
    #                    help='O_C_I_to_M,O_C_to_M, O_I_to_M,C_I_to_M')
    parser.add_argument('--protocol', type=str, default="O_C_I_to_M",
                        help='I_C_M_to_O,I_C_to_O, I_M_to_O,C_M_to_O, O_to_O')
    
    # training settings
    parser.add_argument('--model_type', type=str, default="ResGCN", help='model_type')
    parser.add_argument('--model_name', type=str, default="resnet18", help='model_name')
    parser.add_argument('--model_name1', type=str, default="gcn", help='model_name')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='pretrained')
    parser.add_argument('--eval_preq', type=int, default=1, help='eval_preq size')
    parser.add_argument('--img_size', type=int, default=256, help='img size')

    parser.add_argument('--pretrain', type=str, default='imagenet', help='imagenet')
    parser.add_argument('--num_classes', type=int, default=2, help='num_classes')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    #parser.add_argument('--freeze', type=bool, default=False, help='bn_freeze')
    parser.add_argument('--batch_size', type=int, default=80, help='batch size')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--top_k', type=int, default=25)
    parser.add_argument('--normfc', type=str2bool, default=True)
    parser.add_argument('--train_rotation', type=str2bool, default=True, help='True,False')
    parser.add_argument('--train_extra', type=str2bool, default= False, help='True,False')
    parser.add_argument('--train_scale_min', type=float, default=0.2, help='batch size')
    parser.add_argument('--test_scale', type=float, default=0.9, help='batch size')
    parser.add_argument('--base_lr', type=float, default=0.004, help='base learning rate')
    parser.add_argument('--feat_loss', type=str, default='supcon', help='')
    parser.add_argument('--feat_loss_weight', type=float, default=0.1, help='')
    parser.add_argument('--graph_loss_weight', type=float, default=0.07, help='')
    parser.add_argument('--seed', type=int, default=0, help='batch size')
    parser.add_argument('--temperature', type=float, default=0.1, help='')

    parser.add_argument('--device', type=str, default='1', help='device id, format is like 0,1,2')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--num_epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--print_freq', type=int, default=5, help='print frequency')
    parser.add_argument('--resume', type=bool, default=False, help='print frequency')

    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--trans', type=str, default="p", help="different pre-process")
    # optimizer
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # debug
    parser.add_argument('--debug_subset_size', type=int, default=None)

    parser.add_argument('--scale', type=str, default='1', help='')

    return parser.parse_args()


def str2bool(x):
    return x.lower() in ('true')


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    pretrain_alias = {
        "imagenet": "img",
    }
    args.result_name = f"({args.protocol})_warmup_({args.warmup_epochs})_bsz({args.batch_size})" + \
                   f"_epochs({args.num_epochs})_lr({args.base_lr})_step_size({args.step_size})_gamma({args.gamma})" + \
                   f"_top-k_({args.top_k})_loss-inter({args.feat_loss_weight})_loss-intra({args.graph_loss_weight})"

    info_list = [args.protocol,args.warmup_epochs, args.batch_size, args.num_epochs, args.base_lr,args.gamma, args.feat_loss_weight, args.graph_loss_weight, args.top_k]

    args.summary = "\t".join([str(info) for info in info_list])
    print(args.result_name)
    #print(args.summary)

    if args.protocol == "I_C_M_to_O":
        args.num_epochs *= 1
        args.step_size *= 1

    # 移动模型到 GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args=args)
