import numpy as np


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
        exp_metric = np.array([results[seed][exp][method][metric] for exp in exp_list])
        metric_np[i, :] = exp_metric
    return metric_np


def get_accuracies(results):
    pass


def get_coverage():
    pass


def get_avg_set_sizes():
    pass
