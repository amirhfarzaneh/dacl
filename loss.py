import torch
import torch.nn as nn
from torch.autograd.function import Function


class SparseCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(SparseCenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        self.sparse_centerloss = SparseCenterLossFunction.apply
        self.feat_dim = feat_dim
        self.size_average = size_average
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.centers.data.t())

    def forward(self, feat, A, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.sparse_centerloss(feat, A, label, self.centers, batch_size_tensor)
        return loss


class SparseCenterLossFunction(Function):
    @staticmethod
    def forward(ctx, feature, A, label, centers, batch_size):
        ctx.save_for_backward(feature, A, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (A * (feature - centers_batch).pow(2)).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, A, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = feature - centers_batch
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        # A gradient
        grad_A = diff.pow(2) / 2.0 / batch_size

        counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), - A * diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        return grad_output * A * diff / batch_size, grad_output * grad_A, None, grad_centers, None
