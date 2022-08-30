import numpy as np
import torch


# args = get_args()
# print(args.gpu)
# cuda_str = "cuda:{}".format(str(args.gpu))
# device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")
# print(device)


# ===  Conformal L2D === #
# Quantile calculations ===
# def qhat_conformal(model, x_calib, y_calib, alpha=0.9):
#     n = x_calib.shape[0]
#     # Model softmax output ===
#     out = model(x_calib.to(device).float())
#     # Sort output and indexes
#     # sorted, pi = out.softmax(dim=1).sort(dim=1, descending=True)
#     sorted, pi = out.sort(dim=1, descending=True)
#     # Scores: cumsum until true label ===
#     scores = sorted.cumsum(dim=1).gather(1, pi.argsort(1))[range(y_calib.shape[0]), y_calib]
#     # Get the score quantile ===
#     qhat = torch.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)
#     return qhat

# Quantile Model calculations ===
def qhat_conformal_model(model, x_calib, y_calib, alpha=0.9):
    n = x_calib.shape[0]
    # Compute softmax for K classes, without deferral.
    out_logits = model(x_calib.float(), logits_out=True)
    # Discard deferal category
    out = out_logits[:, :-1].softmax(dim=1)

    sorted, pi = out.sort(dim=1, descending=True)
    scores = sorted.cumsum(dim=1).gather(1, pi.argsort(1))[range(y_calib.shape[0]), y_calib]
    # Get the score quantile
    qhat = torch.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)
    return qhat


def qhat_conformal_model_logits(out_logits, y_calib, alpha=0.9):
    n = len(y_calib)
    # Discard deferal category
    out = out_logits[:, :-1].softmax(dim=1)

    # Scores
    sorted, pi = out.sort(dim=1, descending=True)
    # Add until y_true (y_calib)
    scores = sorted.cumsum(dim=1).gather(1, pi.argsort(1))[range(y_calib.shape[0]), y_calib]

    # Get the score quantile
    qhat = torch.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)
    return qhat


# Quantile Expert calculations ===
def qhat_conformal_expert(model, expert_fn, x_calib, y_calib, device, alpha=0.9, statistic=1):
    n = x_calib.shape[0]
    out = model(x_calib.float())

    sorted, pi = out.sort(dim=1, descending=True)
    defer_dim = out.shape[-1] - 1  # defer dim number

    # Expert correctnes ===
    m = expert_fn(x_calib, y_calib).to(device)
    expert_correct = (m == y_calib)
    # Obtain "stop" condition depending on statistic
    stop_label = torch.zeros(y_calib.shape, dtype=torch.int64).to(device)

    # Statistic 1 ===
    if statistic == 1:
        # expert correct, pi_j = K+1
        stop_label[expert_correct] = defer_dim

        # expert incorrect, pi_j = argmin(y*, K+1) out
        out_truelabel = out[range(y_calib.shape[0]), y_calib]
        out_deferal = out[:, -1]
        true_gt_deferal = out_truelabel > out_deferal
        stop_label[(~expert_correct) & true_gt_deferal] = defer_dim  # true > deferal: set deferal
        stop_label[(~expert_correct) & (~true_gt_deferal)] = y_calib[(~expert_correct) & (~true_gt_deferal)]  # deferal > true: set true

    # Statistic 2 ===
    elif statistic == 2:
        stop_label = defer_dim  # pi_j = K+1

    # Statistic 3 ===
    elif statistic == 3:  # TODO: NOT USE
        stop_label = defer_dim  # pi_j = K+1

        # condition = (expert_correct) & (out[:, -1] > out[range(y_calib.shape[0]), y_calib])
        # stop_label[condition] = defer_dim  # pi_j = K+1
        # stop_label[~condition] = y_calib[~condition]  # pi_j = K+1

    # Get Scores
    scores = sorted.cumsum(dim=1).gather(1, pi.argsort(1))[range(y_calib.shape[0]), stop_label]  # stop label!
    # Get the score quantile
    qhat = torch.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)
    return qhat


def qhat_conformal_expert_logits(out_logits, exp_pred, y_calib, device, alpha=0.9, statistic=1):
    n = len(y_calib)
    out = out_logits.softmax(dim=1)

    sorted, pi = out.sort(dim=1, descending=True)
    defer_dim = out.shape[-1] - 1  # defer dim number

    # Expert correctness ===
    m = exp_pred.to(device)
    expert_correct = (m == y_calib)
    # Obtain "stop" condition depending on statistic
    stop_label = torch.zeros(y_calib.shape, dtype=torch.int64).to(device)

    # Statistic 1 ===
    if statistic == 1:
        # expert correct, pi_j = K+1
        stop_label[expert_correct] = defer_dim

        # expert incorrect, pi_j = argmin(y*, K+1) out
        out_truelabel = out[range(y_calib.shape[0]), y_calib]
        out_deferal = out[:, -1]
        true_gt_deferal = out_truelabel > out_deferal
        stop_label[(~expert_correct) & true_gt_deferal] = defer_dim  # true > deferal: set deferal
        stop_label[(~expert_correct) & (~true_gt_deferal)] = y_calib[(~expert_correct) & (~true_gt_deferal)]  # deferal > true: set true

    # Statistic 2 ===
    elif statistic == 2:
        stop_label = defer_dim  # pi_j = K+1

    # Statistic 3 ===
    elif statistic == 3:  # TODO: NOT USE
        stop_label = defer_dim  # pi_j = K+1

        # condition = (expert_correct) & (out[:, -1] > out[range(y_calib.shape[0]), y_calib])
        # stop_label[condition] = defer_dim  # pi_j = K+1
        # stop_label[~condition] = y_calib[~condition]  # pi_j = K+1

    # Get Scores
    scores = sorted.cumsum(dim=1).gather(1, pi.argsort(1))[range(y_calib.shape[0]), stop_label]  # stop label!
    # Get the score quantile
    qhat = torch.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)
    return qhat


# Prediction Sets ===
# def get_prediction_sets(model, x_test, q_hat):
#     # Deploy (output=list of lenght n, each element is tensor of classes)
#     out = model(x_test.to(device).float())
#     # Softmax
#     # out = out.softmax(dim=1)
#     test_sorted, test_pi = out.sort(dim=1, descending=True)
#     sizes = (test_sorted.cumsum(dim=1) > q_hat).int().argmax(dim=1)
#     prediction_sets = [test_pi[i][:(sizes[i] + 1)] for i in range(sizes.shape[0])]
#     return prediction_sets


# Prediction Sets Model ===
def get_prediction_sets_model(model, x_test, q_hat):
    out_logits = model(x_test.float(), logits_out=True)
    # Discard defedral category
    out = out_logits[:, :-1].softmax(dim=1)

    test_sorted, test_pi = out.sort(dim=1, descending=True)
    sizes = (test_sorted.cumsum(dim=1) > q_hat).int().argmax(dim=1)
    prediction_sets = [test_pi[i][:(sizes[i] + 1)] for i in range(sizes.shape[0])]
    return prediction_sets


def get_prediction_sets_model_logits(out_logits, q_hat):
    # Discard defedral category
    out = out_logits[:, :-1].softmax(dim=1)

    test_sorted, test_pi = out.sort(dim=1, descending=True)
    sizes = (test_sorted.cumsum(dim=1) > q_hat).int().argmax(dim=1)
    prediction_sets = [test_pi[i][:(sizes[i] + 1)] for i in range(sizes.shape[0])]
    return prediction_sets


# Prediction Sets Expert ===
def get_prediction_sets_expert(model, x_test, q_hat, statistic=2):
    # Deploy (output=list of lenght n, each element is tensor of classes)
    out = model(x_test.float())
    defer_dim = out.shape[-1] - 1  # defer dim number

    test_sorted, test_pi = out.sort(dim=1, descending=True)
    sizes = (test_sorted.cumsum(dim=1) > q_hat).int().argmax(dim=1)
    if statistic == 2:
        stop_at_kplus1 = [torch.where(test_pi[i] == defer_dim)[0] for i in range(out.shape[0])]
        prediction_sets = [test_pi[i][:(idx + 1)] for i, idx in enumerate(stop_at_kplus1)]
    else:
        prediction_sets = [test_pi[i][:(sizes[i] + 1)] for i in range(sizes.shape[0])]

    # Drop deferral dimension (K+1)
    prediction_sets = [set_i[set_i != defer_dim] for set_i in prediction_sets]
    return prediction_sets


def get_prediction_sets_expert_logits(out_logits, q_hat, statistic=2):
    defer_dim = out_logits.shape[-1] - 1  # defer dim number
    out = out_logits.softmax(dim=1)

    test_sorted, test_pi = out.sort(dim=1, descending=True)
    if statistic == 2:
        stop_at_kplus1 = [torch.where(test_pi[i] == defer_dim)[0] for i in range(out.shape[0])]
        prediction_sets = [test_pi[i][:(idx + 1)] for i, idx in enumerate(stop_at_kplus1)]
    else:
        sizes = (test_sorted.cumsum(dim=1) > q_hat).int().argmax(dim=1)
        prediction_sets = [test_pi[i][:(sizes[i] + 1)] for i in range(sizes.shape[0])]

    # Drop deferral dimension (K+1)
    prediction_sets = [set_i[set_i != defer_dim] for set_i in prediction_sets]
    return prediction_sets


# Cardinality ===
def set_cardinality(pred_set):
    """
    Calculate cardinality of different sets.
    """
    cardinality = [torch.tensor(set_i.shape) for set_i in pred_set]
    return torch.tensor(cardinality)


# Defer t ===
def defer_to_expert(model_cardinality, expert_cardinality, device):
    r"""
    Calculate defer to expert (rejection) depending on model and expert cardinalities.

    Args:
        model_cardinality:
        expert_cardinality:
        get:

    Returns:
        reject:
    """
    # 1 = Defer
    reject = torch.tensor(
        [model_card_i > expert_card_i for (model_card_i, expert_card_i) in
         zip(model_cardinality, expert_cardinality)]
    )
    # If it is equal -> Standard L2D
    reject_tiebreak = torch.tensor(
        [model_card_i == expert_card_i for (model_card_i, expert_card_i) in
         zip(model_cardinality, expert_cardinality)]
    )
    return reject.to(device), reject_tiebreak.to(device)


# def defer_to_expert(model_cardinality, expert_cardinality, get=True):
#     r"""
#     Calculate defer to expert (rejection) depending on model and expert cardinalities.
#
#     Args:
#         model_cardinality:
#         expert_cardinality:
#         get:
#
#     Returns:
#         reject: 0=No defer (model), 1=Defer (expert), -1=TieBreak-> Standard L2D
#     """
#
#     if not get:  # greater than or equal
#         reject = [model_card_i >= expert_card_i for (model_card_i, expert_card_i) in
#                   zip(model_cardinality, expert_cardinality)]
#     else:  # greater than
#         reject = [model_card_i > expert_card_i for (model_card_i, expert_card_i) in
#                   zip(model_cardinality, expert_cardinality)]
#
#     return torch.tensor(reject)


# ====== Anastasios Conformal ====== #
"""
From https://github.com/aangelopoulos/conformal_classification
"""

# # Conformalize a model with a calibration set.
# # Save it to a file in .cache/modelname
# # The only difference is that the forward method of ConformalModel also outputs a set.
# class ConformalModel(nn.Module):
#     def __init__(self, model, calib_loader, alpha, kreg=None, lamda=None, randomized=True, allow_zero_sets=False,
#                  pct_paramtune=0.3, batch_size=32, lamda_criterion='size'):
#         super(ConformalModel, self).__init__()
#         self.model = model
#         self.alpha = alpha
#         self.randomized = randomized
#         self.allow_zero_sets = allow_zero_sets
#
#         self.num_classes = calib_loader.dataset.dataset.K  # number of classes
#         self.T = torch.Tensor([1.3])  # initialize (1.3 is usually a good value)
#         self.T, calib_logits = platt(self, calib_loader)
#
#         # K_reg and Lambda ===
#         if kreg == None or lamda == None:
#             kreg, lamda, calib_logits = pick_parameters(model, calib_logits, alpha, kreg, lamda, randomized,
#                                                         allow_zero_sets, pct_paramtune, batch_size, lamda_criterion)
#
#         self.penalties = np.zeros((1, self.num_classes))
#         self.penalties[:, kreg:] += lamda
#
#         calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
#
#         # === Q_hat === #
#         self.Qhat = conformal_calibration_logits(self, calib_loader)
#
#     def forward(self, *args, randomized=None, allow_zero_sets=None, **kwargs):
#         if randomized == None:
#             randomized = self.randomized
#         if allow_zero_sets == None:
#             allow_zero_sets = self.allow_zero_sets
#         # Step Forward: Get logits ===
#         logits = self.model(*args, **kwargs)
#         with torch.no_grad():
#             logits_numpy = logits.detach().cpu().numpy()
#             scores = softmax(logits_numpy / self.T.item(), axis=1)  # temperature scaling
#             # Step Forward: Get logits ===
#
#             # Sort logits ===
#             I, ordered, cumsum = sort_sum(scores)
#             # GCQ function ===
#             S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties,
#                     randomized=randomized, allow_zero_sets=allow_zero_sets)
#
#         return logits, S
#
#
# # Computes the conformal calibration
# # NOT USED ===
# def conformal_calibration(cmodel, calib_loader):
#     print("Conformal calibration w/o logits")
#     with torch.no_grad():
#         E = np.array([])
#         for x, targets in tqdm(calib_loader):
#             logits = cmodel.model(x.to(device=device)).detach().cpu().numpy()
#             scores = softmax(logits / cmodel.T.item(), axis=1)
#
#             I, ordered, cumsum = sort_sum(scores)
#
#             E = np.concatenate((E, giq(scores, targets, I=I, ordered=ordered, cumsum=cumsum, penalties=cmodel.penalties,
#                                        randomized=True, allow_zero_sets=True)))
#
#         Qhat = np.quantile(E, 1 - cmodel.alpha, interpolation='higher')
#
#         return Qhat
#
#
# # Temperature scaling
# def platt(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
#     r"""
#     Platt scaling for calibration data w/o logits.
#     Args:
#         cmodel:
#         calib_loader:
#         max_iters:
#         lr:
#         epsilon:
#
#     Returns:
#
#     """
#     print("Begin Platt scaling.")
#     # Save logits so don't need to double compute them
#     logits_dataset = get_logits_targets(cmodel.model, calib_loader)
#     logits_loader = torch.utils.data.DataLoader(logits_dataset, batch_size=calib_loader.batch_size, shuffle=False,
#                                                 pin_memory=True)
#
#     T = platt_logits(cmodel, logits_loader, max_iters=max_iters, lr=lr, epsilon=epsilon)
#
#     print(f"Optimal T={T.item()}")
#     return T, logits_dataset
#
#
# # ====== Conformal ====== #
#
# # ====== Conformal Logits ====== #
# # Precomputed-logit versions of the above functions.
# # For using LAC
# class ConformalModelLogits(nn.Module):
#     def __init__(self, model, calib_loader, alpha, kreg=None, lamda=None, randomized=True, allow_zero_sets=False,
#                  naive=False, LAC=False, pct_paramtune=0.3, batch_size=32, lamda_criterion='size'):
#         super(ConformalModelLogits, self).__init__()
#         self.model = model
#         self.alpha = alpha
#         self.randomized = randomized
#         self.LAC = LAC
#         self.allow_zero_sets = allow_zero_sets
#         self.T = platt_logits(self, calib_loader)
#
#         # K_reg and Lambda ===
#         if (kreg == None or lamda == None) and not naive and not LAC:
#             kreg, lamda, calib_logits = pick_parameters(model, calib_loader.dataset, alpha, kreg, lamda, randomized,
#                                                         allow_zero_sets, pct_paramtune, batch_size, lamda_criterion)
#             # Create dataloader from calib_logits
#             calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
#
#         self.penalties = np.zeros((1, calib_loader.dataset[0][0].shape[0]))
#
#         # === Q_hat === #
#
#         # Q_hat: Not-kreg, Not-naive, not LAC ===
#         if not (kreg == None) and not naive and not LAC:
#             self.penalties[:, kreg:] += lamda
#         self.Qhat = 1 - alpha
#
#         # Q_hat: Not-naive, not LAC ===
#         if not naive and not LAC:
#             self.Qhat = conformal_calibration_logits(self, calib_loader)
#
#         # Q_hat: Not-naive, not LAC ===
#         elif not naive and LAC:
#             gt_locs_cal = np.array(
#                 [np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in calib_loader.dataset])
#             scores_cal = 1 - np.array(
#                 [np.sort(torch.softmax(calib_loader.dataset[i][0] / self.T.item(), dim=0))[::-1][gt_locs_cal[i]] for i
#                  in range(len(calib_loader.dataset))])
#             self.Qhat = np.quantile(scores_cal, np.ceil((scores_cal.shape[0] + 1) * (1 - alpha)) / scores_cal.shape[0])
#
#     def forward(self, logits, randomized=None, allow_zero_sets=None):
#         if randomized == None:
#             randomized = self.randomized
#         if allow_zero_sets == None:
#             allow_zero_sets = self.allow_zero_sets
#
#         with torch.no_grad():
#             logits_numpy = logits.detach().cpu().numpy()
#             scores = softmax(logits_numpy / self.T.item(), axis=1)
#
#             if not self.LAC:
#                 # Sort logits ===
#                 I, ordered, cumsum = sort_sum(scores)
#                 # GCQ function ===
#                 S = gcq(scores, self.Qhat, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties,
#                         randomized=randomized, allow_zero_sets=allow_zero_sets)
#             else:
#                 S = [np.where((1 - scores[i, :]) < self.Qhat)[0] for i in range(scores.shape[0])]
#
#         return logits, S
#
#
# def conformal_calibration_logits(cmodel, calib_loader):
#     print("Conformal calibration with logits")
#     with torch.no_grad():
#         E = np.array([])
#         for logits, targets in calib_loader:
#             logits = logits.detach().cpu().numpy()
#
#             scores = softmax(logits / cmodel.T.item(), axis=1)
#
#             I, ordered, cumsum = sort_sum(scores)
#
#             E = np.concatenate((E, giq(scores, targets, I=I, ordered=ordered, cumsum=cumsum, penalties=cmodel.penalties,
#                                        randomized=True, allow_zero_sets=True)))
#
#         Qhat = np.quantile(E, 1 - cmodel.alpha, interpolation='higher')
#
#         return Qhat
#
#
# def platt_logits(cmodel, calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
#     r"""
#
#     Args:
#         cmodel:
#         calib_loader:
#         max_iters:
#         lr:
#         epsilon:
#
#     Returns:
#
#     """
#     nll_criterion = nn.CrossEntropyLoss().to(device=device)
#
#     T = nn.Parameter(torch.Tensor([1.3]).to(device=device))
#
#     optimizer = optim.SGD([T], lr=lr)
#     for iter in range(max_iters):
#         T_old = T.item()
#         for x, targets in calib_loader:
#             optimizer.zero_grad()
#             x = x.to(device=device)
#             x.requires_grad = True
#             out = x / T
#             loss = nll_criterion(out, targets.long().to(device=device))
#             loss.backward()
#             optimizer.step()
#         if abs(T_old - T.item()) < epsilon:
#             break
#     return T
#
#
# # ====== Conformal Logits ====== #
#
# # ====== Core Functions ====== #
# # CORE CONFORMAL INFERENCE FUNCTIONS
#
# # Generalized conditional quantile function.
# def gcq(scores, tau, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
#     penalties_cumsum = np.cumsum(penalties, axis=1)
#     sizes_base = ((cumsum + penalties_cumsum) <= tau).sum(axis=1) + 1  # 1 - 1001
#     sizes_base = np.minimum(sizes_base, scores.shape[1])  # 1-1000
#
#     if randomized:
#         V = np.zeros(sizes_base.shape)
#         for i in range(sizes_base.shape[0]):
#             V[i] = 1 / ordered[i, sizes_base[i] - 1] * \
#                    (tau - (cumsum[i, sizes_base[i] - 1] - ordered[i, sizes_base[i] - 1]) - penalties_cumsum[
#                        0, sizes_base[i] - 1])  # -1 since sizes_base \in {1,...,1000}.
#
#         sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
#     else:
#         sizes = sizes_base
#
#     if tau == 1.0:
#         sizes[:] = cumsum.shape[1]  # always predict max size if alpha==0. (Avoids numerical error.)
#
#     if not allow_zero_sets:
#         sizes[
#             sizes == 0] = 1  # allow the user the option to never have empty sets (will lead to incorrect coverage if 1-alpha < model's top-1 accuracy
#
#     S = list()
#
#     # Construct S from equation (5)
#     for i in range(I.shape[0]):
#         S = S + [I[i, 0:sizes[i]], ]
#
#     return S
#
#
# # Get the 'p-value'
# def get_tau(score, target, I, ordered, cumsum, penalty, randomized, allow_zero_sets):  # For one example
#     idx = np.where(I == target)
#     tau_nonrandom = cumsum[idx]
#
#     if not randomized:
#         return tau_nonrandom + penalty[0]
#
#     U = np.random.random()
#
#     if idx == (0, 0):
#         if not allow_zero_sets:
#             return tau_nonrandom + penalty[0]
#         else:
#             return U * tau_nonrandom + penalty[0]
#     else:
#         return U * ordered[idx] + cumsum[(idx[0], idx[1] - 1)] + (penalty[0:(idx[1][0] + 1)]).sum()
#
#
# # Gets the histogram of Taus.
# def giq(scores, targets, I, ordered, cumsum, penalties, randomized, allow_zero_sets):
#     """
#         Generalized inverse quantile conformity score function.
#         E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1] such that the correct label enters.
#     """
#     E = -np.ones((scores.shape[0],))
#     for i in range(scores.shape[0]):
#         E[i] = get_tau(scores[i:i + 1, :], targets[i].item(), I[i:i + 1, :], ordered[i:i + 1, :], cumsum[i:i + 1, :],
#                        penalties[0, :], randomized=randomized, allow_zero_sets=allow_zero_sets)
#
#     return E
#
#
# # ====== Core Functions ====== #
#
#
# # ====== Parameter Tuning ====== #
# ### === AUTOMATIC PARAMETER TUNING FUNCTIONS ### ===
# def pick_kreg(paramtune_logits, alpha):
#     gt_locs_kstar = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in paramtune_logits])
#     kstar = np.quantile(gt_locs_kstar, 1 - alpha, interpolation='higher') + 1
#     return kstar
#
#
# def pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets):
#     # Calculate lamda_star
#     best_size = iter(paramtune_loader).__next__()[0][1].shape[0]  # number of classes
#     # Use the paramtune data to pick lamda.  Does not violate exchangeability.
#     for temp_lam in [0.001, 0.01, 0.1, 0.2, 0.5]:  # predefined grid, change if more precision desired.
#         conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam,
#                                                randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
#         top1_avg, top5_avg, cvg_avg, sz_avg = validate(paramtune_loader, conformal_model, print_bool=False)
#         if sz_avg < best_size:
#             best_size = sz_avg
#             lamda_star = temp_lam
#     return lamda_star
#
#
# def pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets,
#                             strata=[[0, 1], [2, 3], [4, 6], [7, 10], [11, 100], [101, 1000]]):
#     # Calculate lamda_star
#     lamda_star = 0
#     best_violation = 1
#     # Use the paramtune data to pick lamda.  Does not violate exchangeability.
#     for temp_lam in [0, 1e-5, 1e-4, 8e-4, 9e-4, 1e-3, 1.5e-3,
#                      2e-3]:  # predefined grid, change if more precision desired.
#         conformal_model = ConformalModelLogits(model, paramtune_loader, alpha=alpha, kreg=kreg, lamda=temp_lam,
#                                                randomized=randomized, allow_zero_sets=allow_zero_sets, naive=False)
#         curr_violation = get_violation(conformal_model, paramtune_loader, strata, alpha)
#         if curr_violation < best_violation:
#             best_violation = curr_violation
#             lamda_star = temp_lam
#     return lamda_star
#
#
# def pick_parameters(model, calib_logits, alpha, kreg, lamda, randomized, allow_zero_sets, pct_paramtune, batch_size,
#                     lamda_criterion):
#     num_paramtune = int(np.ceil(pct_paramtune * len(calib_logits)))
#     paramtune_logits, calib_logits = tdata.random_split(calib_logits,
#                                                         [num_paramtune, len(calib_logits) - num_paramtune])
#     calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
#     paramtune_loader = tdata.DataLoader(paramtune_logits, batch_size=batch_size, shuffle=False, pin_memory=True)
#
#     if kreg == None:
#         kreg = pick_kreg(paramtune_logits, alpha)
#     if lamda == None:
#         if lamda_criterion == "size":
#             lamda = pick_lamda_size(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
#         elif lamda_criterion == "adaptiveness":
#             lamda = pick_lamda_adaptiveness(model, paramtune_loader, alpha, kreg, randomized, allow_zero_sets)
#     return kreg, lamda, calib_logits
#
#
# def get_violation(cmodel, loader_paramtune, strata, alpha):
#     df = pd.DataFrame(columns=['size', 'correct'])
#     for logit, target in loader_paramtune:
#         # compute output
#         output, S = cmodel(logit)  # This is a 'dummy model' which takes logits, for efficiency.
#         # measure accuracy and record loss
#         size = np.array([x.size for x in S])
#         I, _, _ = sort_sum(logit.numpy())
#         correct = np.zeros_like(size)
#         for j in range(correct.shape[0]):
#             correct[j] = int(target[j] in list(S[j]))
#         batch_df = pd.DataFrame({'size': size, 'correct': correct})
#         df = df.append(batch_df, ignore_index=True)
#     wc_violation = 0
#     for stratum in strata:
#         temp_df = df[(df['size'] >= stratum[0]) & (df['size'] <= stratum[1])]
#         if len(temp_df) == 0:
#             continue
#         stratum_violation = abs(temp_df.correct.mean() - (1 - alpha))
#         wc_violation = max(wc_violation, stratum_violation)
#     return wc_violation  # the violation
# # ====== Parameter Tuning ====== #
