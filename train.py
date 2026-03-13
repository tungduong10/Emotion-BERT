import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np, argparse, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime


def plot_training_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    metrics = [('loss', 'Loss'), ('acc', 'Accuracy'), ('fscore', 'Weighted F1')]

    for ax, (key, title) in zip(axes, metrics):
        ax.plot(epochs, history['train_{}'.format(key)], label='train')
        ax.plot(epochs, history['valid_{}'.format(key)], label='valid')
        ax.plot(epochs, history['test_{}'.format(key)], label='test')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def infer_meld_feature_dims(path):
    with open(path, 'rb') as f:
        data = pk.load(f)
    video_text = data[3]
    video_audio = data[7]
    video_visual = data[8]
    train_vid = data[10]
    test_vid = data[11]
    sample_vid = next(iter(train_vid)) if len(train_vid) > 0 else next(iter(test_vid))
    text_dim = int(np.asarray(video_text[sample_vid]).shape[-1])
    audio_dim = int(np.asarray(video_audio[sample_vid]).shape[-1])
    visual_dim = int(np.asarray(video_visual[sample_vid]).shape[-1])
    return text_dim, visual_dim, audio_dim

def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(path='data/meld_multimodal_features.pkl', batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset(path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset(path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, kl_loss, dataloader, epoch, optimizer=None, train=False, gamma_1=1.0, gamma_2=1.0, gamma_3=1.0):
    losses, preds, labels, masks = [], [], [], []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
        kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob = model(textf, visuf, acouf, umask, qmask, lengths)
        
        lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)

        kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[2])
        kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[2])
        kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[2])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[2])
        
        loss = gamma_1 * loss_function(lp_all, labels_, umask) + \
                gamma_2 * (loss_function(lp_1, labels_, umask) + loss_function(lp_2, labels_, umask) + loss_function(lp_3, labels_, umask)) + \
               gamma_3 * (kl_loss(kl_lp_1, kl_p_all, umask) + kl_loss(kl_lp_2, kl_p_all, umask) + kl_loss(kl_lp_3, kl_p_all, umask))

        lp_ = all_prob.view(-1, all_prob.size()[2])

        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)  
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=150, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--meld-pkl-path', default='data/meld_multimodal_features.pkl', help='MELD multimodal feature pickle path')
    parser.add_argument('--resume-checkpoint', default=os.environ.get('SDT_RESUME_CHECKPOINT', ''), help='checkpoint path to resume model weights')
    parser.add_argument('--iter-id', type=int, default=int(os.environ.get('SDT_ITER', '1')), help='outer iteration index (1-based)')
    parser.add_argument('--iter-lr-decay', type=float, default=0.9, help='LR decay factor applied across outer iterations')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='max grad norm; <=0 disables clipping')

    args = parser.parse_args()
    today = datetime.datetime.now()
    run_stamp = today.strftime('%Y%m%d_%H%M%S')
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    valid_rate = 0.1
    feat2dim = {'IS10':1582, 'denseface':342, 'MELD_audio':300, 'IEMOCAP_text':1024}
    if args.Dataset == 'MELD':
        D_text, D_visual, D_audio = infer_meld_feature_dims(args.meld_pkl_path)
    else:
        D_audio = feat2dim['IS10']
        D_visual = feat2dim['denseface']
        D_text = feat2dim['IEMOCAP_text']

    D_m = D_audio + D_visual + D_text

    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1

    print('temp {}'.format(args.temp))
    print('feature dims -> text: {}, visual: {}, audio: {}'.format(D_text, D_visual, D_audio))

    model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
                                        n_classes=n_classes,
                                        hidden_dim=args.hidden_dim,
                                        n_speakers=n_speakers,
                                        dropout=args.dropout)

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    if cuda:
        model.cuda()

    iter_id = max(1, args.iter_id)
    effective_lr = args.lr * (args.iter_lr_decay ** (iter_id - 1))
    print('base lr: {}, iter_id: {}, iter_lr_decay: {}, effective lr: {}'.format(args.lr, iter_id, args.iter_lr_decay, effective_lr))

    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        checkpoint = torch.load(args.resume_checkpoint, map_location='cuda' if cuda else 'cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Loaded checkpoint: {}'.format(args.resume_checkpoint))
    elif args.resume_checkpoint:
        print('Checkpoint not found, training from scratch: {}'.format(args.resume_checkpoint))

    kl_loss = MaskedKLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=effective_lr, weight_decay=args.l2)

    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(path=args.meld_pkl_path,
                                                                    valid=valid_rate,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=valid_rate,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    history = {
        'train_loss': [], 'valid_loss': [], 'test_loss': [],
        'train_acc': [], 'valid_acc': [], 'test_acc': [],
        'train_fscore': [], 'valid_fscore': [], 'test_fscore': []
    }
    os.makedirs('checkpoints', exist_ok=True)
    best_checkpoint_path = os.path.join('checkpoints', '{}_iter{}_best.pt'.format(args.Dataset.lower(), iter_id))

    for e in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(model, loss_function, kl_loss, train_loader, e, optimizer, True)
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(model, loss_function, kl_loss, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, kl_loss, test_loader, e)
        all_fscore.append(test_fscore)
        all_acc.append(test_acc)
        all_loss.append(test_loss)
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        history['test_acc'].append(test_acc)
        history['train_fscore'].append(train_fscore)
        history['valid_fscore'].append(valid_fscore)
        history['test_fscore'].append(test_fscore)

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask = test_label, test_pred, test_mask
            torch.save({
                'epoch': e + 1,
                'args': vars(args),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_fscore': best_fscore,
                'history': history,
            }, best_checkpoint_path)
            print('Saved best checkpoint to: {}'.format(best_checkpoint_path))

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        if (e+1)%10 == 0:
            print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0))
            print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))


    if args.tensorboard:
        writer.close()

    os.makedirs('plots', exist_ok=True)
    plot_path = os.path.join('plots', '{}_training_{}.png'.format(args.Dataset.lower(), run_stamp))
    plot_training_curves(history, plot_path)

    checkpoint_path = os.path.join('checkpoints', '{}_final_{}.pt'.format(args.Dataset.lower(), run_stamp))
    torch.save({
        'epoch': n_epochs,
        'args': vars(args),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_fscore': best_fscore,
        'history': history,
    }, checkpoint_path)

    print('Test performance..')
    print('F-Score: {}'.format(max(all_fscore)))
    print('F-Score-index: {}'.format(all_fscore.index(max(all_fscore)) + 1))
    print('Training plot saved to: {}'.format(plot_path))
    print('Best model saved to: {}'.format(best_checkpoint_path))
    print('Final model saved to: {}'.format(checkpoint_path))
    
    if not os.path.exists("record_{}_{}_{}.pk".format(today.year, today.month, today.day)):
        with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
            pk.dump({}, f)
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day), 'rb') as f:
        record = pk.load(f)
    key_ = 'name_'
    if record.get(key_, False):
        record[key_].append(max(all_fscore))
    else:
        record[key_] = [max(all_fscore)]
    if record.get(key_+'record', False):
        record[key_+'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0))
    else:
        record[key_+'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0)]
    with open("record_{}_{}_{}.pk".format(today.year, today.month, today.day),'wb') as f:
        pk.dump(record, f)

    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
