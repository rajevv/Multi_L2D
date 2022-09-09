import numpy as np
import torch


def qhat_multi(probs, y_true, experts, n_val, alpha=0.95):
    n_classes = max(y_true) + 1
    n_experts = len(experts)

    # 1. Filter validation samples
    experts_val = experts[::-1]  # IMPORTANT! swap to match prob ordering
    experts_val = [exp[:n_val] for exp in experts_val]
    y_true_val = y_true[:n_val]

    # 2. Get deferred samples
    _, predicted = torch.max(probs[:n_val].data, 1)
    r = (predicted >= n_classes - n_experts)

    # === Only on deferred samples !
    experts_val = [np.array(exp)[r] for exp in experts_val]
    y_true_val = np.array(y_true_val)[r]

    # 3. Sort J model outputs for experts
    probs_experts = probs[:n_val, n_classes:]
    probs_experts = probs_experts[r]
    sort, pi = probs_experts.sort(dim=1, descending=True)

    # 4. Check if experts are correct
    correct_exp = (np.array(experts_val) == np.array(y_true_val)).T

    # idx for correct experts: [[0,1,2], [1,2], [], ...]
    correct_exp_idx = [np.where(correct_exp_i)[0] for correct_exp_i in correct_exp]

    # 5. Obtain expert index
    # obtain the last expert to be retrieved. If empty, then add all values.
    # ! indexes are not the real expert index, but the sorted indexes, e.g. [[1, 0 ,2],  [1,0], [], ...]
    pi_corr_exp = [probs_experts[i, corr_exp].sort(descending=True)[1] for i, corr_exp in enumerate(correct_exp)]
    pi_corr_exp_stop = [pi_corr_exp_i[-1] if len(pi_corr_exp_i) != 0 else -1 for pi_corr_exp_i in
                        pi_corr_exp]  # last expert

    # obtain real expert index back, e.g. [2,1,-1,...]
    pi_stop = [correct_exp_idx[i][pi_corr_exp_stop_i] if len(correct_exp_idx[i]) != 0 else -1
               for i, pi_corr_exp_stop_i in enumerate(pi_corr_exp_stop)]

    scores = sort.cumsum(dim=1).gather(1, pi.argsort(1))[range(len(torch.tensor(pi_stop))), torch.tensor(pi_stop)]
    qhat = torch.quantile(scores, np.ceil((n_val + 1) * (1 - alpha)) / n_val)

# args = get_args()
# print(args.gpu)
# cuda_str = "cuda:{}".format(str(args.gpu))
# device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")
# print(device)

# =========================== #
# === Multi Conformal L2D === #
# =========================== #

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

# =========================== #
# === Multi Conformal L2D === #
# =========================== #


# =========================== #
# ====== Conformal L2D ====== #
# =========================== #
# Quantile Model calculations ===
# def qhat_conformal_model(model, x_calib, y_calib, alpha=0.9):
#     n = x_calib.shape[0]
#     # Compute softmax for K classes, without deferral.
#     out_logits = model(x_calib.float(), logits_out=True)
#     # Discard deferal category
#     out = out_logits[:, :-1].softmax(dim=1)
#
#     sorted, pi = out.sort(dim=1, descending=True)
#     scores = sorted.cumsum(dim=1).gather(1, pi.argsort(1))[range(y_calib.shape[0]), y_calib]
#     # Get the score quantile
#     qhat = torch.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)
#     return qhat
#
#
# def qhat_conformal_model_logits(out_logits, y_calib, alpha=0.9):
#     n = len(y_calib)
#     # Discard deferal category
#     out = out_logits[:, :-1].softmax(dim=1)
#
#     # Scores
#     sorted, pi = out.sort(dim=1, descending=True)
#     # Add until y_true (y_calib)
#     scores = sorted.cumsum(dim=1).gather(1, pi.argsort(1))[range(y_calib.shape[0]), y_calib]
#
#     # Get the score quantile
#     qhat = torch.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)
#     return qhat
#
#
# # Quantile Expert calculations ===
# def qhat_conformal_expert(model, expert_fn, x_calib, y_calib, device, alpha=0.9, statistic=1):
#     n = x_calib.shape[0]
#     out = model(x_calib.float())
#
#     sorted, pi = out.sort(dim=1, descending=True)
#     defer_dim = out.shape[-1] - 1  # defer dim number
#
#     # Expert correctnes ===
#     m = expert_fn(x_calib, y_calib).to(device)
#     expert_correct = (m == y_calib)
#     # Obtain "stop" condition depending on statistic
#     stop_label = torch.zeros(y_calib.shape, dtype=torch.int64).to(device)
#
#     # Statistic 1 ===
#     if statistic == 1:
#         # expert correct, pi_j = K+1
#         stop_label[expert_correct] = defer_dim
#
#         # expert incorrect, pi_j = argmin(y*, K+1) out
#         out_truelabel = out[range(y_calib.shape[0]), y_calib]
#         out_deferal = out[:, -1]
#         true_gt_deferal = out_truelabel > out_deferal
#         stop_label[(~expert_correct) & true_gt_deferal] = defer_dim  # true > deferal: set deferal
#         stop_label[(~expert_correct) & (~true_gt_deferal)] = y_calib[
#             (~expert_correct) & (~true_gt_deferal)]  # deferal > true: set true
#
#     # Statistic 2 ===
#     elif statistic == 2:
#         stop_label = defer_dim  # pi_j = K+1
#
#     # Statistic 3 ===
#     elif statistic == 3:  # TODO: NOT USE
#         stop_label = defer_dim  # pi_j = K+1
#
#         # condition = (expert_correct) & (out[:, -1] > out[range(y_calib.shape[0]), y_calib])
#         # stop_label[condition] = defer_dim  # pi_j = K+1
#         # stop_label[~condition] = y_calib[~condition]  # pi_j = K+1
#
#     # Get Scores
#     scores = sorted.cumsum(dim=1).gather(1, pi.argsort(1))[range(y_calib.shape[0]), stop_label]  # stop label!
#     # Get the score quantile
#     qhat = torch.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)
#     return qhat
#
#
# def qhat_conformal_expert_logits(out_logits, exp_pred, y_calib, device, alpha=0.9, statistic=1):
#     n = len(y_calib)
#     out = out_logits.softmax(dim=1)
#
#     sorted, pi = out.sort(dim=1, descending=True)
#     defer_dim = out.shape[-1] - 1  # defer dim number
#
#     # Expert correctness ===
#     m = exp_pred.to(device)
#     expert_correct = (m == y_calib)
#     # Obtain "stop" condition depending on statistic
#     stop_label = torch.zeros(y_calib.shape, dtype=torch.int64).to(device)
#
#     # Statistic 1 ===
#     if statistic == 1:
#         # expert correct, pi_j = K+1
#         stop_label[expert_correct] = defer_dim
#
#         # expert incorrect, pi_j = argmin(y*, K+1) out
#         out_truelabel = out[range(y_calib.shape[0]), y_calib]
#         out_deferal = out[:, -1]
#         true_gt_deferal = out_truelabel > out_deferal
#         stop_label[(~expert_correct) & true_gt_deferal] = defer_dim  # true > deferal: set deferal
#         stop_label[(~expert_correct) & (~true_gt_deferal)] = y_calib[
#             (~expert_correct) & (~true_gt_deferal)]  # deferal > true: set true
#
#     # Statistic 2 ===
#     elif statistic == 2:
#         stop_label = defer_dim  # pi_j = K+1
#
#     # Statistic 3 ===
#     elif statistic == 3:  # TODO: NOT USE
#         stop_label = defer_dim  # pi_j = K+1
#
#         # condition = (expert_correct) & (out[:, -1] > out[range(y_calib.shape[0]), y_calib])
#         # stop_label[condition] = defer_dim  # pi_j = K+1
#         # stop_label[~condition] = y_calib[~condition]  # pi_j = K+1
#
#     # Get Scores
#     scores = sorted.cumsum(dim=1).gather(1, pi.argsort(1))[range(y_calib.shape[0]), stop_label]  # stop label!
#     # Get the score quantile
#     qhat = torch.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)
#     return qhat
#
#
# # Prediction Sets ===
# # def get_prediction_sets(model, x_test, q_hat):
# #     # Deploy (output=list of lenght n, each element is tensor of classes)
# #     out = model(x_test.to(device).float())
# #     # Softmax
# #     # out = out.softmax(dim=1)
# #     test_sorted, test_pi = out.sort(dim=1, descending=True)
# #     sizes = (test_sorted.cumsum(dim=1) > q_hat).int().argmax(dim=1)
# #     prediction_sets = [test_pi[i][:(sizes[i] + 1)] for i in range(sizes.shape[0])]
# #     return prediction_sets
#
#
# # Prediction Sets Model ===
# def get_prediction_sets_model(model, x_test, q_hat):
#     out_logits = model(x_test.float(), logits_out=True)
#     # Discard defedral category
#     out = out_logits[:, :-1].softmax(dim=1)
#
#     test_sorted, test_pi = out.sort(dim=1, descending=True)
#     sizes = (test_sorted.cumsum(dim=1) > q_hat).int().argmax(dim=1)
#     prediction_sets = [test_pi[i][:(sizes[i] + 1)] for i in range(sizes.shape[0])]
#     return prediction_sets
#
#
# def get_prediction_sets_model_logits(out_logits, q_hat):
#     # Discard defedral category
#     out = out_logits[:, :-1].softmax(dim=1)
#
#     test_sorted, test_pi = out.sort(dim=1, descending=True)
#     sizes = (test_sorted.cumsum(dim=1) > q_hat).int().argmax(dim=1)
#     prediction_sets = [test_pi[i][:(sizes[i] + 1)] for i in range(sizes.shape[0])]
#     return prediction_sets
#
#
# # Prediction Sets Expert ===
# def get_prediction_sets_expert(model, x_test, q_hat, statistic=2):
#     # Deploy (output=list of lenght n, each element is tensor of classes)
#     out = model(x_test.float())
#     defer_dim = out.shape[-1] - 1  # defer dim number
#
#     test_sorted, test_pi = out.sort(dim=1, descending=True)
#     sizes = (test_sorted.cumsum(dim=1) > q_hat).int().argmax(dim=1)
#     if statistic == 2:
#         stop_at_kplus1 = [torch.where(test_pi[i] == defer_dim)[0] for i in range(out.shape[0])]
#         prediction_sets = [test_pi[i][:(idx + 1)] for i, idx in enumerate(stop_at_kplus1)]
#     else:
#         prediction_sets = [test_pi[i][:(sizes[i] + 1)] for i in range(sizes.shape[0])]
#
#     # Drop deferral dimension (K+1)
#     prediction_sets = [set_i[set_i != defer_dim] for set_i in prediction_sets]
#     return prediction_sets
#
#
# def get_prediction_sets_expert_logits(out_logits, q_hat, statistic=2):
#     defer_dim = out_logits.shape[-1] - 1  # defer dim number
#     out = out_logits.softmax(dim=1)
#
#     test_sorted, test_pi = out.sort(dim=1, descending=True)
#     if statistic == 2:
#         stop_at_kplus1 = [torch.where(test_pi[i] == defer_dim)[0] for i in range(out.shape[0])]
#         prediction_sets = [test_pi[i][:(idx + 1)] for i, idx in enumerate(stop_at_kplus1)]
#     else:
#         sizes = (test_sorted.cumsum(dim=1) > q_hat).int().argmax(dim=1)
#         prediction_sets = [test_pi[i][:(sizes[i] + 1)] for i in range(sizes.shape[0])]
#
#     # Drop deferral dimension (K+1)
#     prediction_sets = [set_i[set_i != defer_dim] for set_i in prediction_sets]
#     return prediction_sets
#
#
# # Cardinality ===
# def set_cardinality(pred_set):
#     """
#     Calculate cardinality of different sets.
#     """
#     cardinality = [torch.tensor(set_i.shape) for set_i in pred_set]
#     return torch.tensor(cardinality)
#
#
# # Defer t ===
# def defer_to_expert(model_cardinality, expert_cardinality, device):
#     r"""
#     Calculate defer to expert (rejection) depending on model and expert cardinalities.
#
#     Args:
#         model_cardinality:
#         expert_cardinality:
#         get:
#
#     Returns:
#         reject:
#     """
#     # 1 = Defer
#     reject = torch.tensor(
#         [model_card_i > expert_card_i for (model_card_i, expert_card_i) in
#          zip(model_cardinality, expert_cardinality)]
#     )
#     # If it is equal -> Standard L2D
#     reject_tiebreak = torch.tensor(
#         [model_card_i == expert_card_i for (model_card_i, expert_card_i) in
#          zip(model_cardinality, expert_cardinality)]
#     )
#     return reject.to(device), reject_tiebreak.to(device)
#
#
# # def defer_to_expert(model_cardinality, expert_cardinality, get=True):
# #     r"""
# #     Calculate defer to expert (rejection) depending on model and expert cardinalities.
# #
# #     Args:
# #         model_cardinality:
# #         expert_cardinality:
# #         get:
# #
# #     Returns:
# #         reject: 0=No defer (model), 1=Defer (expert), -1=TieBreak-> Standard L2D
# #     """
# #
# #     if not get:  # greater than or equal
# #         reject = [model_card_i >= expert_card_i for (model_card_i, expert_card_i) in
# #                   zip(model_cardinality, expert_cardinality)]
# #     else:  # greater than
# #         reject = [model_card_i > expert_card_i for (model_card_i, expert_card_i) in
# #                   zip(model_cardinality, expert_cardinality)]
# #
# #     return torch.tensor(reject)

# =========================== #
# ====== Conformal L2D ====== #
# =========================== #
