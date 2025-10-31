import os
import shutil
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.backends.cudnn as cudnn
import numpy as np
from json import dump
import random
import math
import re
from glob import glob

# ----------------------------------- training-related -----------------------------------
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    torch.cuda.empty_cache()


def set_device(gpu=None):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        print(f'Available GPUs Index : {os.environ["CUDA_VISIBLE_DEVICES"]}')
    except KeyError:
        print('No GPU available, using CPU ... ')
    return torch.device('cuda') if torch.cuda.device_count() >= 1 else torch.device('cpu')


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def save_params(params, params_file, json_format=False):
    with open(params_file, 'w') as f:
        if not json_format:
            params_file.replace('.json', '.txt')
            for k, v in params.__dict__.items():
                f.write(f'{k:<20}: {v}\n')
        else:
            params_file.replace('.txt', '.json')
            dump(params.__dict__, f, indent=4)


def save_config(params, params_file):
    config_file_path = params.cfg_file
    shutil.copy(config_file_path, params_file)


def save_network_info(model, path):
    with open(path, 'w') as f:
        f.writelines(model.__repr__())



# ----------------------------------- configuration-related -----------------------------------
def str_is_int(x):
    if x.count('-') > 1:
        return False
    if x.isnumeric():
        return True
    if x.startswith('-') and x.replace('-', '').isnumeric():
        return True
    return False


def str_is_float(x):
    if str_is_int(x):
        return False
    try:
        _ = float(x)
        return True
    except ValueError:
        return False


class Config(object):
    def set_item(self, key, value):
        if isinstance(value, str):
            if str_is_int(value):
                value = int(value)
            elif str_is_float(value):
                value = float(value)
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'none':
                value = None
        if key.endswith('milestones'):
            try:
                tmp_v = value[1:-1].split(',')
                value = list(map(int, tmp_v))
            except:
                raise AssertionError(f'{key} is: {value}, format not supported!')
        self.__dict__[key] = value

    def __repr__(self):
        ret = 'Config:\n{\n'
        for k in self.__dict__.keys():
            s = f'    {k}: {self.__dict__[k]}\n'
            ret += s
        ret += '}\n'
        return ret


def load_from_cfg(path):
    # try easydict
    cfg = Config()
    if not path.endswith('.cfg'):
        path = path + '.cfg'
    if not os.path.exists(path) and os.path.exists('config' + os.sep + path):
        path = 'config' + os.sep + path
    assert os.path.isfile(path), f'{path} is not a valid config file.'

    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    for line in lines:
        if line.startswith('['):
            continue
        k, v = line.replace(' ', '').split('=')
        cfg.set_item(key=k, value=v)
    cfg.set_item(key='cfg_file', value=path)

    return cfg


# ----------------------------------- optimization-related -----------------------------------
def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

def lr_warmup(lr_list, lr_init, warmup_end_epoch=5):
    lr_list[:warmup_end_epoch] = list(np.linspace(0, lr_init, warmup_end_epoch))
    return lr_list


def lr_scheduler(lr_init, num_epochs, warmup_end_epoch=5, mode='cosine',
                 epoch_decay_start=None, epoch_decay_ratio=None, epoch_decay_interval=None):
    """

    :param lr_init：initial learning rate
    :param num_epochs: number of epochs
    :param warmup_end_epoch: number of warm up epochs
    :param mode: {cosine, linear, step}
                  cosine:
                        lr_t = 0.5 * lr_0 * (1 + cos(t * pi / T)) in t'th epoch of T epochs
                  linear:
                        lr_t = (T - t) / (T - t_decay) * lr_0, after t_decay'th epoch
                  step:
                        lr_t = lr_0 * ratio**(t//interval), e.g. ratio = 0.1 with interval = 30;
                                                                 ratio = 0.94 with interval = 2
    :param epoch_decay_start: used in linear mode as `t_decay`
    :param epoch_decay_ratio: used in step mode as `ratio`
    :param epoch_decay_interval: used in step mode as `interval`
    :return:
    """
    lr_list = [lr_init] * num_epochs

    print('| Learning rate warms up for {} epochs'.format(warmup_end_epoch))
    lr_list = lr_warmup(lr_list, lr_init, warmup_end_epoch)

    print('| Learning rate decays in {} mode'.format(mode))
    if mode == 'cosine':
        for t in range(warmup_end_epoch, num_epochs):
            lr_list[t] = 0.5 * lr_init * (1 + math.cos((t - warmup_end_epoch + 1) * math.pi /
                                                       (num_epochs - warmup_end_epoch + 1)))
    elif mode == 'linear':
        if type(epoch_decay_start) == int and epoch_decay_start > warmup_end_epoch:
            for t in range(epoch_decay_start, num_epochs):
                lr_list[t] = float(num_epochs - t) / (num_epochs - epoch_decay_start) * lr_init
        else:
            raise AssertionError('Please specify epoch_decay_start, '
                                 'and epoch_decay_start need to be larger than warmup_end_epoch')
    elif mode == 'step':
        if type(epoch_decay_ratio) == float and type(epoch_decay_interval) == int and epoch_decay_interval < num_epochs:
            for t in range(warmup_end_epoch, num_epochs):
                lr_list[t] = lr_init * epoch_decay_ratio**((t - warmup_end_epoch + 1) // epoch_decay_interval)

    return lr_list


# ----------------------------------- model-related -----------------------------------
def init_weights(module, init_method='He'):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, val=0)


def frozen_layer(module):
    for parameters in module.parameters():
        parameters.required_grad = False


def unfrozen_layer(module):
    for parameters in module.parameters():
        parameters.required_grad = True


def load_dp_dict(net, dp_dict_path, device='cpu'):
    """Load DataParallel Model Dict into Non-DataParallel Model

    :param net: model network (non-DataParallel)
    :param dp_dict_path: model state dict (DataParallel model)
    :param device: target device, i.e. gpu or cpu
    :return:
    """
    model_dict = net.state_dict()
    pretrained_dict = torch.load(dp_dict_path)
    pretrained_dict = {k[7:]: v.to(device) for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    return net


# ----------------------------------- others -----------------------------------
def split_set(x, flag):
    # x shape is (N), x is sorted in descending
    if x.shape[0] == 1:
        return -1
    # tmp = (x < flag).nonzero()
    tmp = torch.nonzero(torch.lt(x, flag), as_tuple=False)
    if tmp.shape[0] == 0:
        return -1
    else:
        return tmp[0, 0] - 1


def kl_div(p, q, base=2):
    # p, q is in shape (batch_size, n_classes)
    if base == 2:
        return (p * p.log2() - p * q.log2()).sum(dim=1)
    else:
        return (p * p.log() - p * q.log()).sum(dim=1)


def symmetric_kl_div(p, q, base=2):
    return kl_div(p, q, base) + kl_div(q, p, base)


def js_div(p, q, base=2):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, base) + 0.5 * kl_div(q, m, base)


def entropy(p):
    return Categorical(probs=p).entropy()


def get_smoothed_label_distribution(labels, nc, epsilon):
    smoothed_label = torch.full(size=(labels.size(0), nc), fill_value=epsilon / (nc - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    return smoothed_label.to(labels.device)


def get_stats(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
    test_acc_list = []
    test_acc_list2 = []
    valid_epoch = []
    # valid_epoch = [191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    for idx in range(1, 11):
        line = lines[-idx].strip()
        epoch, train_loss, train_acc, test_loss, test_acc = line.split(' | ')[:5]
        ep = int(epoch.split(': ')[1])
        valid_epoch.append(ep)
        # assert ep in valid_epoch, ep
        if '/' not in test_acc:
            test_acc_list.append(float(test_acc.split(': ')[1]))
        else:
            test_acc1, test_acc2 = map(lambda x: float(x), test_acc.split(': ')[1].lstrip('(').rstrip(')').split('/'))
            test_acc_list.append(test_acc1)
            test_acc_list2.append(test_acc2)
    if len(test_acc_list2) == 0:
        test_acc_list = np.array(test_acc_list)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.2f}, std: {test_acc_list.std():.2f}')
        return {'mean': test_acc_list.mean(), 'std': test_acc_list.std(), 'valid_epoch': valid_epoch}
    else:
        test_acc_list = np.array(test_acc_list)
        test_acc_list2 = np.array(test_acc_list2)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.2f}, std: {test_acc_list.std():.2f}')
        print(f'mean: {test_acc_list2.mean():.2f}, std: {test_acc_list2.std():.2f}')
        return {'mean1': test_acc_list.mean(), 'std1': test_acc_list.std(),
                'mean2': test_acc_list2.mean(), 'std2': test_acc_list2.std(),
                'valid_epoch': valid_epoch}
def read_last_finished_epoch(logfile_path: str) -> int:
    """
    从 log.txt 里解析出最后完成的 epoch（基于你 logger.info 的行）
    返回已完成的 epoch 数（0 表示还没开始）。
    """
    if not os.path.isfile(logfile_path):
        return 0
    last_epoch = 0
    try:
        with open(logfile_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 匹配：epoch:  12 | train loss: ...
                m = re.search(r'epoch:\s*(\d+)\s*\|', line)
                if m:
                    last_epoch = int(m.group(1))
    except Exception:
        pass
    return last_epoch

def auto_find_latest_run_with_ckpt(root_dir: str, exclude_dir: str | None = None) -> str | None:
    """
    在 root_dir 下找 *包含至少一个 .pth* 的最近 run 目录。
    会排除 exclude_dir（通常是当前新建的 result_dir）。
    """
    if not os.path.isdir(root_dir):
        return None
    subdirs = [p for p in glob(os.path.join(root_dir, '*')) if os.path.isdir(p)]
    if not subdirs:
        return None

    # 排除当前 run 目录
    if exclude_dir is not None:
        subdirs = [p for p in subdirs if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(exclude_dir))]
    if not subdirs:
        return None

    # 只保留有检查点的
    def has_ckpt(d):
        return bool(glob(os.path.join(d, 'best_epoch.pth'))) or bool(glob(os.path.join(d, 'epoch_*.pth')))
    subdirs = [d for d in subdirs if has_ckpt(d)]
    if not subdirs:
        return None

    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subdirs[0]

def auto_pick_ckpt(run_dir: str) -> str | None:
    """
    优先 best_epoch.pth；否则选择编号最大的 epoch_*.pth。
    """
    if not run_dir:
        return None
    best = os.path.join(run_dir, 'best_epoch.pth')
    if os.path.isfile(best):
        return best
    cks = glob(os.path.join(run_dir, 'epoch_*.pth'))
    if not cks:
        return None
    def _epoch_num(p):
        m = re.search(r'epoch_(\d+)\.pth$', os.path.basename(p))
        return int(m.group(1)) if m else -1
    cks.sort(key=_epoch_num, reverse=True)
    return cks[0]