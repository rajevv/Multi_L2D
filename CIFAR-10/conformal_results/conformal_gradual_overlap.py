import sys

sys.path.append("../")  # append for conformal function
from conformal import conformal


# n_val = int(0.8 * len(true[0]))
# n_test = len(true[0]) - n_val
# print("N val:{}".format(n_val))
# print("N test:{}".format(n_test))
#
# # Method dict ===
# method_list_ova = ["standard", "last", "random", "voting"]
# method_dict_ova = {"standard": [],
#                    "last": [],
#                    "random": [],
#                    "voting": []}
#
# for method in method_list_ova:
#
#     print("Method: {}\n".format(method))
#     for i, p in enumerate(p_out):
#         # =============
#         # = Get Probs =
#         # =============
#
#         probs = confs[i]
#         experts = exps[i]
#         experts = experts[::-1]  # reverse order!
#         y_true = true[-i]
#
#         # Val/Calibration ===
#         probs_val = probs[:n_val, n_classes:]
#         experts_val = [exp[:n_val] for exp in experts]
#         y_true_val = y_true[:n_val]
#
#         # Test ===
#         probs_test = probs[n_val:, n_classes:]
#         experts_test = [exp[n_val:] for exp in experts]
#         y_true_test = y_true[n_val:]
#
#         # =============
#         # = Conformal =
#         # =============
#
#         # Calculate Q_hat ===
#
#         # === Only on deferred samples
#         _, predicted = torch.max(probs[:n_val].data, 1)
#         r = (predicted >= n_classes_exp - n_experts)
#
#         # Filter
#         probs_experts = probs_val[r]
#         experts_val = [np.array(exp)[r] for exp in experts_val]
#         y_true_val = np.array(y_true_val)[r]
#
#         # Model expert probs ===
#         # Sort J model outputs for experts
#         sort, pi = probs_experts.sort(dim=1, descending=True)
#
#         # Correctness experts ===
#         # Check if experts are correct
#         correct_exp = (np.array(experts_val) == np.array(y_true_val)).T
#         # idx for correct experts: [[0,1,2], [1,2], [], ...]
#         correct_exp_idx = [np.where(correct_exp_i)[0] for correct_exp_i in correct_exp]
#
#         # obtain the last expert to be retrieved. If empty, then add all values.
#         # indexes are not the real expert index, but the sorted indexes, e.g. [[1, 0 ,2],  [1,0], [], ...]
#         pi_corr_exp = [probs_experts[i, corr_exp].sort(descending=True)[1] for i, corr_exp in enumerate(correct_exp)]
#         pi_corr_exp_stop = [pi_corr_exp_i[-1] if len(pi_corr_exp_i) != 0 else -1 for pi_corr_exp_i in
#                             pi_corr_exp]  # last expert
#
#         # obtain real expert index back, e.g. [2,1,-1,...]
#         pi_stop = [correct_exp_idx[i][pi_corr_exp_stop_i] if len(correct_exp_idx[i]) != 0 else -1 for
#                    i, pi_corr_exp_stop_i in enumerate(pi_corr_exp_stop)]
#
#         scores = sort.cumsum(dim=1).gather(1, pi.argsort(1))[range(len(torch.tensor(pi_stop))), torch.tensor(pi_stop)]
#         n_quantile = r.sum()
#         qhat = torch.quantile(scores, np.ceil((n_quantile + 1) * (1 - alpha)) / n_quantile, interpolation="higher")
#
#         print("Q_hat {}: {}".format(p, qhat))
#
#         # =============
#         # = Metrics =
#         # =============
#
#         # === Initalize ====
#
#         correct = 0
#         correct_sys = 0
#         exp = 0
#         exp_total = 0
#         total = 0
#         real_total = 0
#         alone_correct = 0
#
#         # Individual Expert Accuracies === #
#         expert_correct_dic = {k: 0 for k in range(len(experts_test))}
#         expert_total_dic = {k: 0 for k in range(len(experts_test))}
#
#         probs_test_exp = probs_test
#         probs_test_model = probs[n_val:]
#
#         # Predicted value
#         _, predicted = torch.max(probs_test_model.data, 1)
#
#         # Classifier alone prediction
#         _, prediction = torch.max(probs_test_model.data[:, :(n_classes_exp - n_experts)], 1)
#
#         labels = y_true_test
#         for i in range(0, n_test):
#             r = (predicted[i].item() >= n_classes_exp - n_experts)
#             alone_correct += (prediction[i] == labels[i]).item()
#
#             # Non-deferred
#             if r == 0:
#                 total += 1
#                 correct += (predicted[i] == labels[i]).item()
#                 correct_sys += (predicted[i] == labels[i]).item()
#
#             # Deferred
#             if r == 1:
#                 # Non Conformal prediction ===
#                 if method == "standard":
#                     deferred_exp = (predicted[i] - n_classes).item()  # reverse order, as in loss function
#                     exp_prediction = experts_test[deferred_exp][i]
#
#                     # Conformal prediction ===
#                 else:
#                     # Sort J model outputs for experts. sorted probs and sorted indexes
#                     sort_i, pi_i = probs_test_exp[i].sort(descending=True)
#                     # Get last sorted index to be below Q_hat
#                     pi_stop_i = (sort_i.cumsum(dim=0) <= qhat).sum()
#                     # Prediction sets
#                     prediction_set_i = (pi_i[:(pi_stop_i)]).numpy()  # not allow empty sets
#
#                     # - Get expert prediction depending on method
#                     # ======
#                     exp_prediction = get_expert_prediction(experts_test, prediction_set_i, method=method)
#                     # ======
#
#                 # Deferral accuracy: No matter expert ===
#                 exp += (exp_prediction == labels[i])
#                 exp_total += 1
#                 # Individual Expert Accuracy ===
#                 # expert_correct_dic[deferred_exp] += (exp_prediction == labels[i].item())
#                 # expert_total_dic[deferred_exp] += 1
#                 #
#                 correct_sys += (exp_prediction == labels[i])
#
#             real_total += 1
#
#         #  ===  Coverage  === #
#         cov = 100 * total / real_total
#
#         #  === Individual Expert Accuracies === #
#         expert_accuracies = {"expert_{}".format(str(k)): 100 * expert_correct_dic[k] / (expert_total_dic[k] + 0.0002)
#                              for k
#                              in range(len(experts_test))}
#
#         # Add expert accuracies dict
#         to_print = {"coverage": cov,
#                     "system_accuracy": 100 * correct_sys / real_total,
#                     "expert_accuracy": 100 * exp / (exp_total + 0.0002),
#                     "classifier_accuracy": 100 * correct / (total + 0.0001),
#                     "alone_classifier": 100 * alone_correct / real_total}
#         print(to_print, flush=True)
#
#         # Save to method dict ===
#         method_dict_ova[method].append(to_print)
#

#
if __name__ == '__main__':

    # =========== #
    # === OvA === #
    # =========== #
    # Load data OvA ===
    ova_path = "../ova_gradual_overlap/"
    path_confidence_ova = ova_path + "confidence_multiple_experts"
    path_experts_ova = ova_path + "expert_predictions_multiple_experts"
    path_labels_ova = ova_path + "true_label_multiple_experts"
    model_name = "_p_out_{}"  # to include values in exp_list
    exp_list = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    seeds = [436, 625, 948]

    ova_results = conformal.load_results(path_confidence_ova, path_experts_ova, path_labels_ova, model_name,
                                         seeds, exp_list, method="ova")

    # Process Results ===
    exp_args = {"n_experts": 10,
                "n_classes": 10,
                "ensemble_size": 5}
    conformal.process_conformal_results(ova_results, exp_list, exp_args, cal_percent=0.8, alpha=0.1)


    # =============== #
    # === Softmax === #
    # =============== #
    # Load data Softmax ===
    softmax_path = "../softmax_gradual_overlap/"
    path_confidence_softmax = softmax_path + "confidence_multiple_experts"
    path_experts_softmax = softmax_path + "expert_predictions_multiple_experts"
    path_labels_softmax = softmax_path + "true_label_multiple_experts"
    model_name = "_p_out_{}"  # to include values in exp_list
    exp_list = [0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
    seeds = [436, 625, 948]

    softmax_results = conformal.load_results(path_confidence_softmax, path_experts_softmax, path_labels_softmax,
                                             model_name, seeds, exp_list, method="softmax")

