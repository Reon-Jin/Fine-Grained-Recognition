from __future__ import print_function
import os
import torch
import torch.utils.data
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from multiprocessing import freeze_support
from tqdm import tqdm
from torch import amp

# 从 config 中引入超参数和数据接口
from config import (
    BATCH_SIZE,
    PROPOSAL_NUM,
    SAVE_FREQ,
    LR,
    WD,
    resume,
    save_dir as BASE_SAVE_DIR,
    get_train_dataset,
    get_val_dataset
)
from core import model
from core.utils import init_log


def main():
    # 仅使用 GPU 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark = True

    # 准备保存目录
    start_epoch = 1
    save_dir = os.path.join(BASE_SAVE_DIR, datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

    # 构建数据集与 DataLoader
    trainset = get_train_dataset()
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    valset = get_val_dataset()
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # 定义模型
    net = model.attention_net(topN=PROPOSAL_NUM)
    if resume:
        ckpt = torch.load(resume, map_location='cuda:0')
        net.load_state_dict(ckpt['net_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    # 将模型移到 GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scaler = amp.GradScaler()

    # 定义各分支优化器
    raw_optimizer = torch.optim.SGD(
        net.pretrained_model.parameters(), lr=LR, momentum=0.9, weight_decay=WD
    )
    part_optimizer = torch.optim.SGD(
        net.proposal_net.parameters(), lr=LR, momentum=0.9, weight_decay=WD
    )
    concat_optimizer = torch.optim.SGD(
        net.concat_net.parameters(), lr=LR, momentum=0.9, weight_decay=WD
    )
    partcls_optimizer = torch.optim.SGD(
        net.partcls_net.parameters(), lr=LR, momentum=0.9, weight_decay=WD
    )

    # 学习率调度
    schedulers = [
        MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
        MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
        MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
        MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)
    ]

    best_val_acc = 0.0
    final_epoch = None

    # 训练循环
    for epoch in range(start_epoch, 500):
        _print('--' * 50)

        # 训练阶段
        net.train()
        for img, label in tqdm(trainloader, desc=f'Epoch {epoch:03d} Training', ncols=100):
            img, label = img.to(device), label.to(device)
            batch_size = img.size(0)

            raw_optimizer.zero_grad()
            part_optimizer.zero_grad()
            concat_optimizer.zero_grad()
            partcls_optimizer.zero_grad()
            with amp.autocast(device_type='cuda'):
                raw_logits, concat_logits, part_logits, _, top_n_prob = net(img)

                # 计算各分支损失
                part_loss = model.list_loss(
                    part_logits.view(batch_size * PROPOSAL_NUM, -1),
                    label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)
                ).view(batch_size, PROPOSAL_NUM)
                raw_loss = criterion(raw_logits, label)
                concat_loss = criterion(concat_logits, label)
                rank_loss = model.ranking_loss(top_n_prob, part_loss)
                partcls_loss = criterion(
                    part_logits.view(batch_size * PROPOSAL_NUM, -1),
                    label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)
                )

                total_loss = raw_loss + rank_loss + concat_loss + partcls_loss

            scaler.scale(total_loss).backward()

            scaler.step(raw_optimizer)
            scaler.step(part_optimizer)
            scaler.step(concat_optimizer)
            scaler.step(partcls_optimizer)
            scaler.update()

        # 验证与保存
        if epoch % SAVE_FREQ == 0 or epoch == 499:
            net.eval()

            # 评估验证集
            val_loss, val_correct, total = 0.0, 0, 0
            for img, label in tqdm(val_loader, desc=f'Epoch {epoch:03d} Eval Val', ncols=100):
                img, label = img.to(device), label.to(device)
                with torch.no_grad(), amp.autocast(device_type='cuda'):
                    _, concat_logits, _, _, _ = net(img)
                    loss = criterion(concat_logits, label)
                    _, pred = concat_logits.max(1)

                batch_size = label.size(0)
                total += batch_size
                val_correct += pred.eq(label).sum().item()
                val_loss += loss.item() * batch_size

            val_acc = val_correct / total
            val_loss /= total
            _print(f'epoch:{epoch} - val loss: {val_loss:.3f} acc: {val_acc:.3f}')

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'net_state_dict': net.state_dict()
                }, os.path.join(save_dir, 'best.ckpt'))

            # 保存最后一轮模型
            if epoch == 499:
                torch.save({
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'net_state_dict': net.state_dict()
                }, os.path.join(save_dir, 'final.ckpt'))

        # 调度学习率
        for scheduler in schedulers:
            scheduler.step()

    print('finishing training')


if __name__ == '__main__':
    freeze_support()
    main()
