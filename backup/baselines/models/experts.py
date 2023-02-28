# Experts for CIFAR-10
from __future__ import division

import random

import numpy as np

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
num_classes = len(class_names)
class2idx = {class_name: idx for idx, class_name in enumerate(class_names)}
idx2class = {idx: class_name for class_name, idx in class2idx.items()}


# Synthetic Expert for Non-overlapping expertise
class synth_expert2:
    def __init__(self, k1=None, k2=None, n_classes=None, S=None, p_in=None, p_out=None):
        '''
        class to model the non-overlapping synthetic experts

        The expert predicts correctly for classes k1 (inclusive) to k2 (exclusive), and
        random across the total number of classes for other classes outside of [k1, k2).

        For example, an expert could be correct for classes 2 (k1) to 4 (k2) for CIFAR-10.

        '''
        self.k1 = k1
        self.k2 = k2
        self.p_in = p_in if p_in is not None else 1.0
        self.p_out = p_out if p_out is not None else 1 / n_classes
        self.n_classes = n_classes
        self.S = S  # list : set of classes where the oracle predicts

    # expert correct in [k1,k2) classes else random across all the classes
    def predict(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][0].item() < self.k2 and labels[i][0].item() >= self.k1:
                outs[i] = labels[i][0].item()
            else:
                prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
        return outs

    # expert correct in [k1, k2) classes with prob. p_in; correct on other classes with prob. p_out
    def predict_prob_cifar(self, input, labels):
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][0].item() < self.k2 and labels[i][0].item() >= self.k1:
                coin_flip = np.random.binomial(1, self.p_in)
                if coin_flip == 1:
                    outs[i] = labels[i][0].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes - 1)
            else:
                coin_flip = np.random.binomial(1, self.p_out)
                if coin_flip == 1:
                    outs[i] = labels[i][0].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes - 1)
        return outs

    def predict_prob_cifar_2(self, input, labels):
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][0].item() in self.S:
                coin_flip = np.random.binomial(1, self.p_in)
                if coin_flip == 1:
                    outs[i] = labels[i][0].item()
                if coin_flip == 0:
                    outs[i] = random.choice(list(set(range(self.n_classes)) - set(self.S)))
            else:
                coin_flip = np.random.binomial(1, self.p_out)
                if coin_flip == 1:
                    outs[i] = labels[i][0].item()
                if coin_flip == 0:
                    outs[i] = random.choice(list(set(range(self.n_classes)) - set(self.S)))
        return outs


class synth_expert:
    '''
    simple class to describe our synthetic expert on CIFAR-10
    ----
    k: number of classes expert can predict
    n_classes: number of classes (10+1 for CIFAR-10)
    '''

    def __init__(self, k, n_classes, p_in=1, p_out=0.2):
        self.k = k
        self.n_classes = n_classes
        self.p_in = p_in
        self.p_out = p_out if p_out is not None else 1 / self.n_classes

    def predict(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][0].item() <= self.k:
                outs[i] = labels[i][0].item()
            else:
                prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
        return outs

    def predict(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][0].item() <= self.k:
                outs[i] = labels[i][0].item()
            else:
                prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
        return outs

    def predict_biasedK(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][0].item() <= self.k:
                coin_flip = np.random.binomial(1, 0.7)
                if coin_flip == 1:
                    outs[i] = labels[i][0].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes - 1)
            else:
                prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
        return outs

    def predict_prob_cifar(self, input, labels):
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][0].item() <= self.k:
                coin_flip = np.random.binomial(1, self.p_in)
                if coin_flip == 1:
                    outs[i] = labels[i][0].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes - 1)
            else:
                coin_flip = np.random.binomial(1, self.p_out)
                if coin_flip == 1:
                    outs[i] = labels[i][0].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes - 1)
        return outs

    def predict_prob(self, input, labels, p1=0.75, p2=0.20):
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][0].item() <= self.k:
                coin_flip = np.random.binomial(1, p1)
                if coin_flip == 1:
                    outs[i] = labels[i][0].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes - 1)
            else:
                coin_flip = np.random.binomial(1, p2)
                if coin_flip == 1:
                    outs[i] = labels[i][0].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes - 1)
        return outs

    def predict_biased(self, input, labels):
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            coin_flip = np.random.binomial(1, 0.7)
            if coin_flip == 1:
                outs[i] = labels[i][0].item()
            if coin_flip == 0:
                outs[i] = random.randint(0, self.n_classes - 1)
        return outs

    def predict_random(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            prediction_rand = random.randint(0, self.n_classes - 1)
            outs[i] = prediction_rand
        return outs

    # expert which only knows k labels and even outside of K predicts randomly from K
    def predict_severe(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][0].item() <= self.k:
                outs[i] = labels[i][0].item()
            else:
                prediction_rand = random.randint(0, self.k)
                outs[i] = prediction_rand
        return outs

    # when the input is OOD, expert predicts correctly else not
    def oracle(self, input, labels):
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][1].item() == 0:
                outs[i] = labels[i][0]
            else:
                if labels[i][0].item() <= self.k:
                    outs[i] = labels[i][0].item()
                else:
                    prediction_rand = random.randint(0, self.n_classes - 1)
                    outs[i] = prediction_rand
        return outs


# ================= #
# Hemmer et all experts
# ================= #

# CIFAR10
class CIFAR10Expert:

    def __init__(self, expert_classes, p_in=0.7):
        self.p_in = p_in
        self.expert_classes = expert_classes
        self.expert_classes_idx = [class2idx[cls] for cls in self.expert_classes]
        self.n_classes = num_classes

    def predict(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][0].item() in self.expert_classes_idx:
                coin_flip = np.random.binomial(1, self.p_in)
                if coin_flip == 1:
                    outs[i] = labels[i][0].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes - 1)
            else:
                prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
        return outs

    def predict_prob(self, input, labels, p1=0.75, p2=0.20):
        batch_size = labels.size()[0]
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i][0].item() <= self.k:
                coin_flip = np.random.binomial(1, p1)
                if coin_flip == 1:
                    outs[i] = labels[i][0].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes - 1)
            else:
                coin_flip = np.random.binomial(1, p2)
                if coin_flip == 1:
                    outs[i] = labels[i][0].item()
                if coin_flip == 0:
                    outs[i] = random.randint(0, self.n_classes - 1)
        return outs

    def predict_random(self, input, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            prediction_rand = random.randint(0, self.n_classes - 1)
            outs[i] = prediction_rand
        return outs

# class Cifar10Expert(synth_expert):
#     """A class used to represent an Expert on CIFAR100 data.
#
#     Parameters
#     ----------
#     strengths : list[int]
#         list of subclass indices defining the experts strengths
#     weaknesses : list[int]
#         list of subclass indices defining the experts weaknesses
#
#     Attributes
#     ----------
#     strengths : list[int]
#         list of subclass indices defining the experts strengths. If the subclass index of an image is in strength the expert makes a correct prediction, if not expert predicts a random superclass
#     weaknesses : list[int]
#         list of subclass indices defining the experts weaknesses. If the subclass index of an image is in weaknesses the expert predicts a random superclass, if not the expert makes a correct prediction
#     use_strengths : bool
#         a boolean indicating whether the expert is defined by its strengths or its weaknesses. True if strengths are not empty, False if strengths are empty
#     subclass_idx_to_superclass_idx : dict of {int : int}
#         a dictionary that maps the 100 subclass indices of CIFAR100 to their 20 superclass indices
#
#     Methods
#     -------
#     predict(fine_ids)
#         makes a prediction based on the specified strengths or weaknesses and the given subclass indices
#     """
#
#     # Hemmer et al code
#     def __init__(self, k, n_classes, p_in=1, p_out=0.2):
#         super(Cifar10Expert, self).__init__(k, n_classes, p_in, p_out)
#
#     # def __init__(self, strengths: list = [], weaknesses: list = []):
#     #     self.strengths = strengths
#     #     self.weaknesses = weaknesses
#     #
#     #     assert len(self.strengths) > 0 or len(
#     #         self.weaknesses) > 0, "the competence of a Cifar100Expert needs to be specified using either strengths or weaknesses"
#     #
#     #     self.use_strengths = len(self.strengths) > 0
#     #
#     #     self.subclass_idx_to_superclass_idx = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3,
#     #                                            11: 14, 12: 9, 13: 18, 14: 7,
#     #                                            15: 11, 16: 3, 17: 9, 18: 7, 19: 11,
#     #                                            20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3,
#     #                                            29: 15, 30: 0, 31: 11, 32: 1,
#     #                                            33: 10, 34: 12, 35: 14, 36: 16, 37: 9,
#     #                                            38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14,
#     #                                            47: 17, 48: 18, 49: 10, 50: 16,
#     #                                            51: 4, 52: 17, 53: 4, 54: 2, 55: 0,
#     #                                            56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12,
#     #                                            65: 16, 66: 12, 67: 1, 68: 9,
#     #                                            69: 19, 70: 2, 71: 10, 72: 0, 73: 1,
#     #                                            74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 19, 82: 2,
#     #                                            83: 4, 84: 6, 85: 19, 86: 5,
#     #                                            87: 5, 88: 8, 89: 19, 90: 18, 91: 1,
#     #                                            92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}
#
#     # def predict(self, subclass_idxs: list) -> list:
#     #     """Predicts the superclass indices for the given subclass indices
#     #
#     #     Parameters
#     #     ----------
#     #     subclass_idxs : list of int
#     #         list of subclass indices to get a prediction for. Predictions are made based on the specified strengths or weaknesses.
#     #
#     #     Returns
#     #     -------
#     #     list of int
#     #         returns a list of superclass indices that represent the experts prediction
#     #
#     #     """
#     #     predictions = []
#     #     if self.use_strengths:
#     #         for subclass_idx in subclass_idxs:
#     #             if subclass_idx in self.strengths:
#     #                 predictions.append(self.subclass_idx_to_superclass_idx[subclass_idx.item()])
#     #             else:
#     #                 predictions.append(random.randint(0, 19))
#     #     else:
#     #         for subclass_idx in subclass_idxs:
#     #             if subclass_idx in self.weaknesses:
#     #                 predictions.append(random.randint(0, 19))
#     #             else:
#     #                 predictions.append(self.subclass_idx_to_superclass_idx[subclass_idx.item()])
#     #
#     #     return predictions
