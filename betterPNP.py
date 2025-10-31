import os
import sys
import pathlib
import time
import datetime
import argparse
import shutil
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.cuda.amp import autocast, GradScaler
from utils.core import accuracy, evaluate
from utils.builder import *
from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger, print_to_logfile, print_to_console
from utils.loss import *
from utils.module import MLPHead
from PIL import ImageFile
from torchvision.models import efficientnet_v2_l
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

if not hasattr(np, "int"):
    np.int = int

LOG_FREQ = 1

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class CLDataTransform(object):
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w1 = self.transform_weak(sample)
        x_w2 = self.transform_weak(sample)
        x_s = self.transform_strong(sample)
        return x_w1, x_w2, x_s


# ---------- 辅助函数 ----------
def init_weights(m, init_method='He'):
    if isinstance(m, nn.Linear):
        if init_method.lower() in ['he', 'kaiming']:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class EfficientnetV2L(nn.Module):
    def __init__(self, num_classes=200, pretrained=False,
                 activation='tanh',local_weights_path=None):
        super().__init__()

        # ===== backbone =====
        self.backbone = efficientnet_v2_l(weights=None)

        # ===== 载入预训练权重 =====
        if pretrained and local_weights_path is not None:
            sd = torch.load(local_weights_path, map_location="cpu")
            if "state_dict" in sd:  # Lightning / timm 格式兼容
                sd = sd["state_dict"]
            new_sd = {}
            for k, v in sd.items():
                new_k = k[len("module."):] if k.startswith("module.") else k
                new_sd[new_k] = v
            try:
                self.backbone.load_state_dict(new_sd, strict=True)
            except Exception:
                self.backbone.load_state_dict(new_sd, strict=False)

        # ===== 基础特征提取层 =====
        self.features = self.backbone.features
        self.feat_dim = getattr(self.backbone.classifier[-1], 'in_features', 1280)
        self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # ===== 多头分类器 =====
        # Head 1: Linear
        self.classifier_linear = nn.Linear(self.feat_dim, num_classes)
        nn.init.kaiming_normal_(self.classifier_linear.weight, nonlinearity="relu")

        # Head 2: MLP-1
        self.classifier_mlp1 = MLPHead(
            self.feat_dim, mlp_scale_factor=2, projection_size=num_classes,
            init_method='He', activation='relu'
        )

        # Head 3: MLP-2
        self.classifier_mlp2 = MLPHead(
            self.feat_dim, mlp_scale_factor=3, projection_size=num_classes,
            init_method='Xavier', activation='tanh'
        )

        act = 'leaky relu' if activation == 'l_relu' else activation
        self.proba_head = nn.Sequential(
            MLPHead(self.feat_dim, mlp_scale_factor=2, projection_size=3,
                    init_method='He', activation=act),
            nn.Sigmoid(),
        )

        self.classifier_type = "multi-head"
        self.attention = ECA(channels=1280)

    def forward(self, x):
        N = x.size(0)
        feat_map = self.features(x)

        feat_map = self.attention(feat_map)

        feat = self.neck(feat_map).view(N, -1)

        # === 三个头各自输出 ===
        logits_linear = self.classifier_linear(feat)
        logits_mlp1 = self.classifier_mlp1(feat)
        logits_mlp2 = self.classifier_mlp2(feat)

        # === 融合平均 ===
        logits = (logits_linear + logits_mlp1 + logits_mlp2) / 3.0

        prob = self.proba_head(feat)
        return {
            'logits': logits,
            'prob': prob
        }

def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)


def record_network_arch(result_dir, net):
    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net.__repr__())


def get_smoothed_label_distribution(labels, num_class, epsilon):
    smoothed_label = torch.full(size=(labels.size(0), num_class), fill_value=epsilon / (num_class - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    return smoothed_label.to(labels.device)


def build_logger(params):
    logger_root = f'Results/{params.dataset}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(logger_root, "Better_PNP_efficientnet_v2L", params.project, f'{params.log}-{logtime}')
    logger = Logger(logging_dir=result_dir, DEBUG=True)
    logger.set_logfile(logfile_name='log.txt')
    # save_config(params, f'{result_dir}/params.cfg')
    save_params(params, f'{result_dir}/params.json', json_format=True)
    save_current_script(result_dir)
    logger.msg(f'Result Path: {result_dir}')
    return logger, result_dir

def build_model_optim_scheduler(params, device, build_scheduler=True):
    assert not params.dataset.startswith('cifar')
    n_classes = params.n_classes

    net = EfficientnetV2L(
        num_classes=n_classes,
        pretrained=True,  # 如需加载本地权重则为 True，否则 False
        local_weights_path="weight/efficientnet_v2_l_imagenet1k_v1.pth"  # 如有本地权重，取消注释
    )
    if params.opt == 'sgd':
        optimizer = build_sgd_optimizer(net.parameters(), params.lr, params.weight_decay, nesterov=True)
    elif params.opt == 'adam':
        optimizer = build_adam_optimizer(net.parameters(), params.lr)
    elif params.opt == 'adamw':
        optimizer = build_adamw_optimizer(net.parameters(), params.lr, params.weight_decay)
    else:
        raise AssertionError(f'{params.opt} optimizer is not supported yet.')
    if build_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3,
                                                               verbose=True, threshold=1e-4)
    else:
        scheduler = None
    return net.to(device), optimizer, scheduler, n_classes


def build_lr_plan(params, factor=10, decay='linear'):
    lr_plan = [params.lr] * params.epochs
    for i in range(0, params.warmup_epochs):
        lr_plan[i] *= factor
    for i in range(params.warmup_epochs, params.epochs):
        if decay == 'linear':
            lr_plan[i] = float(params.epochs - i) / (params.epochs - params.warmup_epochs) * params.lr  # linearly decay
        elif decay == 'cosine':
            lr_plan[i] = 0.5 * params.lr * (1 + math.cos(
                (i - params.warmup_epochs + 1) * math.pi / (params.epochs - params.warmup_epochs + 1)))  # cosine decay
    return lr_plan


def build_dataset_loader(params):

    transform = build_transform(rescale_size=params.rescale_size, crop_size=params.crop_size)

    dataset = build_webfg_dataset(os.path.join(params.database, params.dataset),
                                    CLDataTransform(transform['train'], transform['train_strong_aug']),
                                    transform['test'])

    train_loader = DataLoader(dataset['train'], batch_size=params.batch_size, shuffle=True, num_workers=8,
                              pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=False)
    return dataset, train_loader, test_loader


def wrapup_training(result_dir, best_accuracy):
    stats = get_stats(f'{result_dir}/log.txt')
    with open(f'{result_dir}/result_stats.txt', 'w') as f:
        f.write(f"valid epochs: {stats['valid_epoch']}\n")
        if 'mean' in stats.keys():
            f.write(f"mean: {stats['mean']:.4f}, std: {stats['std']:.4f}\n")
        else:
            f.write(f"mean1: {stats['mean1']:.4f}, std2: {stats['std1']:.4f}\n")
            f.write(f"mean2: {stats['mean2']:.4f}, std2: {stats['std2']:.4f}\n")
    os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.4f}')

def main(cfg, device):
    init_seeds(0)
    cfg.use_fp16 = False if device.type == 'cpu' else cfg.use_fp16

    logger, result_dir = build_logger(cfg)
    net, optimizer, scheduler, n_classes = build_model_optim_scheduler(cfg, device, build_scheduler=False)
    lr_plan = build_lr_plan(cfg, factor=cfg.warmup_lr_scale, decay=cfg.lr_decay)
    dataset, train_loader, test_loader = build_dataset_loader(cfg)


    logger.msg(
        f"Categories: {n_classes}, Training Samples: {dataset['n_train_samples']}, Testing Samples: {dataset['n_test_samples']}")
    logger.msg(f'Optimizer: {cfg.opt}')
    record_network_arch(result_dir, net)

    if cfg.loss_func_aux == 's-mae':
        aux_loss_func = F.smooth_l1_loss
    elif cfg.loss_func_aux == 'mae':
        aux_loss_func = F.l1_loss
    elif cfg.loss_func_aux == 'mse':
        aux_loss_func = F.mse_loss
    else:
        raise AssertionError(f'{cfg.loss_func_aux} loss is not supported for auxiliary loss yet.')

    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    epoch_train_time = AverageMeter()
    best_accuracy, best_epoch = 0.0, None
    scaler = GradScaler()
    iters_to_accumulate = round(64 / cfg.batch_size) if cfg.use_grad_accumulate and cfg.batch_size < 64 else 1
    logger.msg(
        f'Accumulate gradients every {iters_to_accumulate} iterations --> Acutal batch size is {cfg.batch_size * iters_to_accumulate}')

    entropy_normalize_factor = entropy(torch.ones(n_classes) / n_classes).item()
    # training ---------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(0, cfg.epochs):
        start_time = time.time()

        net.train()
        adjust_lr(optimizer, lr_plan[epoch])
        optimizer.zero_grad()
        train_loss.reset()
        train_accuracy.reset()

        # train this epoch
        pbar = tqdm(train_loader, ncols=150, ascii=' >', leave=False, desc='training')
        for it, sample in enumerate(pbar):


            curr_lr = [group['lr'] for group in optimizer.param_groups][0]

            s = time.time()

            x, x_w, x_s = sample['data']
            x, x_w, x_s = x.to(device), x_w.to(device), x_s.to(device)
            y = sample['label'].to(device)

            with autocast(cfg.use_fp16):
                output = net(x)
                logits = output['logits']
                probs = logits.softmax(dim=1)
                train_acc = accuracy(logits, y, topk=(1,))

                logits_s = net(x_s)['logits']
                logits_w = net(x_w)['logits']

                type_prob = output['prob'].softmax(dim=1)  # (N, 3)
                clean_pred_prob = type_prob[:, 0]
                idn_pred_prob = type_prob[:, 1]
                ood_pred_prob = type_prob[:, 2]

                pbar.set_postfix_str(f'TrainAcc: {train_accuracy.avg:3.2f}%; TrainLoss: {train_loss.avg:3.2f}')
                given_labels = get_smoothed_label_distribution(y, n_classes, epsilon=cfg.epsilon)
                if epoch < cfg.warmup_epochs:
                    pbar.set_description(f'WARMUP TRAINING (lr={curr_lr:.3e})')
                    loss = 0.5 * cross_entropy(logits, given_labels, reduction='mean') + 0.5 * cross_entropy(logits_w,
                                                                                                             given_labels,
                                                                                                             reduction='mean')

                else:
                    pbar.set_description(f'ROBUST TRAINING (lr={curr_lr:.3e})')
                    probs_w = logits_w.softmax(dim=1)
                    with torch.no_grad():
                        mean_pred_prob_dist = (probs + probs_w + given_labels) / 3
                        sharpened_target_s = (mean_pred_prob_dist / cfg.temperature).softmax(dim=1)
                        flattened_target_s = (mean_pred_prob_dist * cfg.temperature).softmax(dim=1)

                    # classification loss
                    loss_clean = 0.5 * cross_entropy(logits, given_labels, reduction='none') + 0.5 * cross_entropy(logits_w, given_labels, reduction='none')
                    loss_idn = cross_entropy(logits_s, sharpened_target_s, reduction='none')
                    loss_ood = cross_entropy(logits_s, flattened_target_s, reduction='none') * cfg.beta

                    # entropy loss
                    loss_entropy = 0.5 * entropy_loss(logits, reduction='none') + 0.5 * entropy_loss(logits_w,reduction='none')
                    loss_clean += loss_entropy * cfg.alpha
                    # consistency loss
                    loss_cons = symmetric_kl_div(probs, probs_w)
                    loss_cls = loss_clean * clean_pred_prob + loss_idn * idn_pred_prob + loss_ood * ood_pred_prob
                    if cfg.neg_cons:
                        loss_cons = loss_cons * (clean_pred_prob + idn_pred_prob - ood_pred_prob)
                    else:
                        loss_cons = loss_cons * (clean_pred_prob + idn_pred_prob)
                    loss_cons = loss_cons.mean()

                    loss_cls = loss_cls.mean()

                    loss = loss_cls + cfg.omega * loss_cons

            scaler.scale(loss).backward()

            if (it + 1) % iters_to_accumulate == 0:
                try:
                    scaler.step(optimizer)
                except RuntimeError:
                    logger.msg('Runtime Error occured! Have unscaled losses and clipped grads before optimizing!')
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2, norm_type=2.0)
                    scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_accuracy.update(train_acc[0], x.size(0))
            train_loss.update(loss.item(), x.size(0))
            epoch_train_time.update(time.time() - s, 1)
            if (cfg.log_freq is not None and (it + 1) % cfg.log_freq == 0) or (it + 1 == len(train_loader)):
                total_mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
                mem = torch.cuda.memory_reserved() / 2 ** 30
                console_content = f"Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  " \
                                  f"Iter:[{it + 1:>4d}/{len(train_loader):>4d}]  " \
                                  f"Train Accuracy:[{train_accuracy.avg:6.2f}]  " \
                                  f"Loss:[{train_loss.avg:4.4f}]  " \
                                  f"GPU-MEM:[{mem:6.3f}/{total_mem:6.3f} Gb]  " \
                                  f"{epoch_train_time.avg:6.2f} sec/iter"
                logger.debug(console_content)

        # evaluate this epoch
        eval_result = evaluate(test_loader, net, device)
        test_accuracy = eval_result['accuracy']
        test_loss = eval_result['loss']
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            if cfg.save_model:
                torch.save(net.state_dict(), f'{result_dir}/best_epoch.pth')
        if cfg.save_model:
            torch.save(net.state_dict(), f"{result_dir}/epoch_{epoch+1}.pth")
        runtime = time.time() - start_time
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'train loss: {train_loss.avg:>6.4f} | '
                    f'train accuracy: {train_accuracy.avg:>6.3f} | '
                    f'test loss: {test_loss:>6.4f} | '
                    f'test accuracy: {test_accuracy:>6.3f} | '
                    f'epoch runtime: {runtime:6.2f} sec | '
                    f'best accuracy: {best_accuracy:6.3f} @ epoch: {best_epoch:03d}')

    wrapup_training(result_dir, best_accuracy)

if __name__ == '__main__':
    # 固定参数配置
    class Config:
        # 数据与模型
        dataset = "web-400"  # 必须以 web- 开头，这样才会走 build_webfg_dataset
        database = "Datasets"  # 上级目录，里面有 WebFG-400/train 和 WebFG-400/val
        n_classes = 400
        classifier = "mlp-1"
        activation = "relu"

        # 训练参数
        batch_size = 16
        lr = 3e-5
        weight_decay = 1e-5
        opt = "adamw"
        epochs = 50
        warmup_epochs = 5
        warmup_lr_scale = 10.0
        lr_decay = "cosine"
        epsilon = 0.5

        # loss 权重
        alpha = 1.0  # entropy loss
        beta = 1.0  # ood classification loss
        omega = 1.0  # consistency loss

        loss_func_aux = "mae"
        weighting = "soft"
        neg_cons = False

        # 其它设置
        rescale_size = 480
        crop_size = 480
        save_model = True
        use_fp16 = True
        use_grad_accumulate = True

        # 日志
        project = "webfg400"
        log = "3heads-ECA-3407"
        log_freq = LOG_FREQ

        # consistency 超参
        temperature = 0.1
        eta = 1.0


    params = Config()
    dev = set_device("0")  # 固定用 GPU:0
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(
        f'Runtime of this script {pathlib.Path(__file__)} : {script_runtime:.1f} seconds ({script_runtime / 3600:.3f} hours)')
