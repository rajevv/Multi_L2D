import numpy as np
import torch
import torch.nn as nn


class Criterion(object):
    def __init__(self):
        pass

    def softmax(self, outputs, labels, collection_Ms, n_classes):
        '''
        The L_{CE} loss implementation for CIFAR
        ----
        outputs: network outputs
        m: cost of deferring to expert cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
        labels: target
        m2:  cost of classifier predicting (alpha* I_{m\neq y} + I_{m =y})
        n_classes: number of classes
        '''
        batch_size = outputs.size()[0]  # batch_size
        m2 = collection_Ms[0][1]
        rcs = []
        for i, _ in enumerate(collection_Ms, 0):
            rcs.append([n_classes-(i+1)] * batch_size)

        temp = -m2 * torch.log2(outputs[range(batch_size), labels])
        for i, (m, _) in enumerate(collection_Ms):
            temp -= m * \
                torch.log2(outputs[range(batch_size), rcs[len(rcs)-i-1]])
        return torch.sum(temp) / batch_size

    def ova(self, outputs, labels, collection_Ms, n_classes):
        '''
        Implementation of OvA surrogate loss for L2D compatible with multiple experts
        outputs : Network outputs (logits! Not softmax values)
        labels : target
        collection_Ms : list of tuple, m vector for each expert
        n_classes : Number of classes (K+E) K=classes in the data, E=number of experts

        '''
        num_experts = len(collection_Ms)
        batch_size = outputs.size()[0]
        l1 = Criterion.LogisticLoss(outputs[range(batch_size), labels], 1)
        l2 = torch.sum(Criterion.LogisticLoss(outputs[:, :(
            n_classes - len(collection_Ms))], -1), dim=1) - Criterion.LogisticLoss(outputs[range(batch_size), labels], -1)

        l3 = 0
        for j in range(num_experts):
            l3 += Criterion.LogisticLoss(
                outputs[range(batch_size), n_classes-1-j], -1)

        l4 = 0
        for j in range(num_experts):
            l4 += collection_Ms[len(collection_Ms)-1-j][0] * (Criterion.LogisticLoss(outputs[range(
                batch_size), n_classes-1-j], 1) - Criterion.LogisticLoss(outputs[range(batch_size), n_classes-1-j], -1))

        l = l1 + l2 + l3 + l4
        return torch.mean(l)

    @staticmethod
    def LogisticLoss(outputs, y):
        outputs[torch.where(outputs == 0.0)] = (-1*y)*(-1*np.inf)
        l = torch.log2(1 + torch.exp((-1*y)*outputs))
        return l
