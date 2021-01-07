import os
import sys
import errno
import torch
import random
import numbers

from sklearn.metrics import f1_score, recall_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels

from torchvision.transforms import functional as F


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target)
        acc = correct.float().sum().mul_(1.0 / batch_size)
    return acc, pred


def calc_metrics(y_pred, y_true, y_scores):
    metrics = {}
    y_pred = torch.cat(y_pred).cpu().numpy()
    y_true = torch.cat(y_true).cpu().numpy()
    y_scores = torch.cat(y_scores).cpu().numpy()
    classes = unique_labels(y_true, y_pred)

    # recall score
    metrics['rec'] = recall_score(y_true, y_pred, average='macro')

    # f1 score
    f1_scores = f1_score(y_true, y_pred, average=None, labels=unique_labels(y_pred))
    metrics['f1'] = f1_scores.sum() / classes.shape[0]

    # AUC PR
    Y = label_binarize(y_true, classes=classes.astype(int).tolist())
    metrics['aucpr'] = average_precision_score(Y, y_scores, average='macro')

    # AUC ROC
    metrics['aucroc'] = roc_auc_score(Y, y_scores, average='macro')

    return metrics


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    console = sys.stdout

    def __init__(self, fpath=None):
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.file is not None:
            self.file.close()


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class RandomFiveCrop(object):

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        # randomly return one of the five crops
        return F.five_crop(img, self.size)[random.randint(0, 4)]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
