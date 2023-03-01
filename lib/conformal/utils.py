import numpy as np
import json

METHOD_METRIC = ['deferral', 'idx_cal', 'coverage_cal', 'coverage_test', 'qhat', 'lamhat']


# Json functions ===
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_dict_as_txt(dict, path):
    with open(path, 'w') as f:
        json.dump(json.dumps(dict, cls=NumpyEncoder), f)


def load_dict_txt(path):
    with open(path, 'r') as f:
        dict = json.loads(json.load(f))
    return dict


def save_metric(results, metric, method_list, path):
    # Retrieve metric
    # Sanity check
    method_list_cp = method_list.copy()
    if metric == "avg_set_size":
        method_list_cp.remove("standard")
        method_list_cp.remove("ensemble")
    # Lists
    seeds_list = list(results.keys())
    exp_list = list(results[seeds_list[0]].keys())
    metric_dict_ova = {method: get_metric(results, seeds_list, exp_list, metric, method)
                       for method in method_list_cp}
    save_dict_as_txt(metric_dict_ova, path)


def get_metric(results, seeds_list, exp_list, metric, method):
    r"""
    Obtain the desired metric from the results dict.
    Args:
        results: Dictionary with seed->exp->method->metric
        seeds_list: List with the available seeds
        exp_list: List with the experiment values, e.g. p_out = [0.1, 0.2,...]
        metric: Desired metric:
            - system_accuracy
            - expert_accuracy
            - classifier_accuracy
            - alone_classifier
        method:

    Returns:

    """

    metric_np = np.zeros((len(seeds_list), len(exp_list)))
    for i, seed in enumerate(seeds_list):
        if metric in METHOD_METRIC:
            exp_metric = np.array([results[seed][exp][metric] for exp in exp_list])
        else:  # implement methods: 'standard', 'last', 'random', 'voting', 'ensemble'
            exp_metric = np.array([results[seed][exp][method][metric] for exp in exp_list])
        metric_np[i, :] = exp_metric
    return metric_np

# def get_accuracies(results):
#     pass
#
#
# def get_coverage():
#     pass
#
#
# def get_avg_set_sizes():
#     pass
