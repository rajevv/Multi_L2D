import sys

sys.path.append("../")  # append for conformal function
from conformal import conformal

if __name__ == '__main__':
    # Process Results ===
    experiment_args = {"n_experts": 10,
                       "n_classes": 10,
                       "ensemble_size": 5}

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
    ova_metrics = conformal.process_conformal_results(ova_results, exp_list, experiment_args, cal_percent=0.8,
                                                      alpha=0.1)

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
    # Process Results ===
    softmax_metrics = conformal.process_conformal_results(softmax_results, exp_list, experiment_args, cal_percent=0.8,
                                                          alpha=0.1)


    print()