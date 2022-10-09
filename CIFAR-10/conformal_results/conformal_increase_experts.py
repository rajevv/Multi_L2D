import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy import stats





n_classes = 10



# === OvA ===
confs = []
exps = []
true = []
path = "ova_increase_experts/"

n_experts = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20]
for n in n_experts:
    model_name = '_' + str(n) + '_experts'
    with open(path + 'confidence_multiple_experts_new' + model_name + '.txt', 'r') as f:
        conf = json.loads(json.load(f))
    with open(path + 'expert_predictions_multiple_experts_new' + model_name + '.txt', 'r') as f:
        exp_pred = json.loads(json.load(f))
    with open(path + 'true_label_multiple_experts_new' + model_name + '.txt', 'r') as f:
        true_label = json.loads(json.load(f))
    true.append(true_label['test'])
    exps.append(exp_pred['test'])
    c = torch.tensor(conf['test'])
    # DANI Correction ===
    c = c.sigmoid()
    # DANI Correction ===
    confs.append(c)

# In[37]:


print("Experts: \n{}".format(np.array(exps[5])[:, 1]))

# ### Check dimensions

# In[52]:


# 1 expert rando, 3 with prob 0.95 correct
probs = confs[-3]
experts = exps[-3]
# experts = experts[::-1]  # reverse order!
y_true = true[-3]

# In[53]:


print("Experts: \n{}".format(np.array(experts)))

# In[54]:


print("Y True: \n{}".format(y_true[:5]))

# In[55]:


n_val = int(0.8 * len(y_true))
n_test = len(y_true) - n_val
print("N val:{}".format(n_val))
print("N test:{}".format(n_test))

# # Conformal Q_hat Calculation

# In[10]:


n_classes_exp = n_classes + n_experts
probs_val = probs[:n_val, 10:]
experts_val = experts
experts_val = [exp[:n_val] for exp in experts_val]

y_true_val = y_true[:n_val]

# === Only on deferred samples
_, predicted = torch.max(probs[:n_val].data, 1)
r = (predicted >= n_classes_exp - n_experts)

# Filter
probs_val = probs_val[r]
experts_val = [np.array(exp)[r] for exp in experts_val]
y_true_val = np.array(y_true_val)[r]

# Model expert probs ===
# Sort J model outputs for experts
probs_experts = probs[:n_val, 10:]
probs_experts = probs_experts[r]
sort, pi = probs_experts.sort(dim=1, descending=True)

# Correctness experts ===
# Check if experts are correct
correct_exp = (np.array(experts_val) == np.array(y_true_val)).T
# idx for correct experts: [[0,1,2], [1,2], [], ...]
correct_exp_idx = [np.where(correct_exp_i)[0] for correct_exp_i in correct_exp]

# obtain the last expert to be retrieved. If empty, then add all values.
# indexes are not the real expert index, but the sorted indexes, e.g. [[1, 0 ,2],  [1,0], [], ...]
pi_corr_exp = [probs_experts[i, corr_exp].sort(descending=True)[1] for i, corr_exp in enumerate(correct_exp)]
pi_corr_exp_stop = [pi_corr_exp_i[-1] if len(pi_corr_exp_i) != 0 else -1 for pi_corr_exp_i in
                    pi_corr_exp]  # last expert

# obtain real expert index back, e.g. [2,1,-1,...]
pi_stop = [correct_exp_idx[i][pi_corr_exp_stop_i] if len(correct_exp_idx[i]) != 0 else -1 for i, pi_corr_exp_stop_i in
           enumerate(pi_corr_exp_stop)]

# =========
n_val = n_val
alpha = 0.1
scores = sort.cumsum(dim=1).gather(1, pi.argsort(1))[range(len(torch.tensor(pi_stop))), torch.tensor(pi_stop)]
qhat = torch.quantile(scores, np.ceil((r.sum() + 1) * (1 - alpha)) / r.sum(), interpolation="higher")

qhat

# # Test

# In[11]:


probs_test = probs[n_val:, n_classes:]
experts_test = [exp[n_val:] for exp in experts]
y_true_test = y_true[n_val:]

# In[12]:


# === Only on deferred samples
_, predicted = torch.max(probs[n_val:].data, 1)
r_test = (predicted >= n_classes_exp - n_experts)

# Filter
probs_test = probs_test[r_test]
experts_test = [np.array(exp)[r_test] for exp in experts_test]
y_true_test = np.array(y_true_test)[r_test]

# Sort J model outputs for experts. sorted probs and sorted indexes
sort_test, pi_test = probs_test.sort(dim=1, descending=True)
# Get last sorted index to be below Q_hat
pi_stop = (sort_test.cumsum(dim=1) <= qhat).sum(axis=1)

# Prediction sets
prediction_sets = [(pi_test[i][:(pi_stop[i])]).numpy() for i in range(pi_stop.shape[0])]  # not allow empty sets
prediction_sets[:5]

# In[13]:


a = np.array([len(prediction_sets_i) for prediction_sets_i in prediction_sets])
import seaborn as sns

sns.histplot(a)

# In[ ]:


# In[ ]:


# In[ ]:


# # Metrics Computation

# ### Accuracy w/o Conformal on deferred samples

# In[14]:


correct = 0
correct_sys = 0
exp = 0
exp_total = 0
total = 0
real_total = 0
alone_correct = 0
#  === Individual Expert Accuracies === #
expert_correct_dic = {k: 0 for k in range(len(experts_test))}
expert_total_dic = {k: 0 for k in range(len(experts_test))}

# In[15]:


# 1 expert rando, 3 with prob 0.9 correct
probs = confs[-1]
experts = exps[-1]
experts = experts[::-1]
y_true = true[-1]

# In[16]:


n_classes = 10
n_experts = 4
n_classes_exp = n_classes + n_experts

probs_test = probs[n_val:]
experts_test = [exp[n_val:] for exp in experts]
y_true_test = y_true[n_val:]

# In[17]:


# Predicted value
_, predicted = torch.max(probs_test.data, 1)
# Classifier alone prediction
_, prediction = torch.max(probs_test.data[:, :(n_classes_exp - n_experts)], 1)


# ### w Conformal Prediction

# In[18]:


def get_expert_prediction(experts, prediction_set_i, method="voting"):
    ensemble_expert_pred_i = np.array(experts_test)[prediction_set_i][:, i]
    if method == "voting":
        exp_prediction = stats.mode(ensemble_expert_pred_i).mode if len(ensemble_expert_pred_i) != 0 else []

    if method == "last":
        exp_prediction = ensemble_expert_pred_i[-1] if len(ensemble_expert_pred_i) != 0 else []

    if method == "random":
        idx = np.random.randint(len(ensemble_expert_pred_i)) if len(ensemble_expert_pred_i) != 0 else -1
        exp_prediction = ensemble_expert_pred_i[idx] if idx != -1 else []

    return exp_prediction


# In[19]:


method = "last"

# In[20]:


labels = y_true_test

# Predicted value
_, predicted = torch.max(probs_test.data, 1)
# Classifier alone prediction
_, prediction = torch.max(probs_test.data[:, :(n_classes_exp - n_experts)], 1)

for i in range(0, n_test):
    r = (predicted[i].item() >= n_classes_exp - len(experts_test))
    alone_correct += (prediction[i] == labels[i]).item()

    # Non-deferred
    if r == 0:
        total += 1
        correct += (predicted[i] == labels[i]).item()
        correct_sys += (predicted[i] == labels[i]).item()

    # Deferred
    if r == 1:
        # Conformal prediction ===
        # Sort J model outputs for experts. sorted probs and sorted indexes
        sort_i, pi_i = probs_test[i, n_classes:].sort(descending=True)
        # Get last sorted index to be below Q_hat
        pi_stop_i = (sort_i.cumsum(dim=0) <= qhat).sum()

        # Prediction sets
        prediction_set_i = (pi_i[:(pi_stop_i)]).numpy()  # not allow empty sets

        # - Get expert prediction depending on method
        # ======
        exp_prediction = get_expert_prediction(experts_test, prediction_set_i, method=method)
        # ======

        # Deferral accuracy: No matter expert ===
        exp += (exp_prediction == labels[i])
        exp_total += 1
        # Individual Expert Accuracy ===
        # expert_correct_dic[deferred_exp] += (exp_prediction == labels[i].item())
        # expert_total_dic[deferred_exp] += 1
        #
        correct_sys += (exp_prediction == labels[i])
    real_total += 1

#  ===  Coverage  === #
cov = str(total) + str(" out of") + str(real_total)

#  === Individual Expert Accuracies === #
expert_accuracies = {"expert_{}".format(str(k)): 100 * expert_correct_dic[k] / (expert_total_dic[k] + 0.0002) for k
                     in range(len(experts_test))}
# Add expert accuracies dict
to_print = {"coverage": cov,
            "system_accuracy": 100 * correct_sys / real_total,
            "expert_accuracy": 100 * exp / (exp_total + 0.0002),
            "classifier_accuracy": 100 * correct / (total + 0.0001),
            "alone_classifier": 100 * alone_correct / real_total}
print(to_print, flush=True)

# In[21]:


print("Cov: {}".format(100 * (total / real_total)))

# In[22]:


expert_accuracies

# In[23]:


100 * exp / (exp_total + 0.0002),

# In[24]:


exp_total

# In[25]:


100 * correct_sys / real_total

# ### w/o Conformal Prediction

# In[26]:


correct = 0
correct_sys = 0
exp = 0
exp_total = 0
total = 0
real_total = 0
alone_correct = 0
#  === Individual Expert Accuracies === #
expert_correct_dic = {k: 0 for k in range(len(experts_test))}
expert_total_dic = {k: 0 for k in range(len(experts_test))}

# In[27]:


# 1 expert rando, 3 with prob 0.9 correct
probs = confs[-3]
experts = exps[-3]
experts = experts[::-1]
y_true = true[-3]

# In[28]:


n_classes = 10
n_experts = 4
n_classes_exp = n_classes + n_experts

probs_test = probs[n_val:]
experts_test = [exp[n_val:] for exp in experts]
y_true_test = y_true[n_val:]

# In[29]:


# Predicted value
_, predicted = torch.max(probs_test.data, 1)
# Classifier alone prediction
_, prediction = torch.max(probs_test.data[:, :(n_classes_exp - n_experts)], 1)

# In[30]:


labels = y_true_test

# Predicted value
_, predicted = torch.max(probs_test.data, 1)
# Classifier alone prediction
_, prediction = torch.max(probs_test.data[:, :(n_classes_exp - n_experts)], 1)
for i in range(0, n_test):
    r = (predicted[i].item() >= n_classes_exp - len(experts_test))
    alone_correct += (prediction[i] == labels[i]).item()
    if r == 0:
        total += 1
        correct += (predicted[i] == labels[i]).item()
        correct_sys += (predicted[i] == labels[i]).item()

    if r == 1:
        deferred_exp = (predicted[i] - n_classes).item()  # reverse order, as in loss function
        # deferred_exp = ((n_classes - 1) - predicted[i]).item()  # reverse order, as in loss function
        exp_prediction = experts_test[deferred_exp][i]
        #
        # Deferral accuracy: No matter expert ===
        exp += (exp_prediction == labels[i])
        exp_total += 1
        # Individual Expert Accuracy ===
        expert_correct_dic[deferred_exp] += (exp_prediction == labels[i])
        expert_total_dic[deferred_exp] += 1
        #
        correct_sys += (exp_prediction == labels[i])
    real_total += 1

#  ===  Coverage  === #
cov = str(total) + str(" out of") + str(real_total)

#  === Individual Expert Accuracies === #
expert_accuracies = {"expert_{}".format(str(k)): 100 * expert_correct_dic[k] / (expert_total_dic[k] + 0.0002) for k
                     in range(len(experts_test))}

# In[31]:


print("Cov: {}".format(100 * (total / real_total)))

# In[32]:


expert_accuracies

# In[33]:


100 * exp / (exp_total + 0.0002),

# In[34]:


exp

# In[35]:


100 * correct_sys / real_total


# # Metric Calculation

# In[58]:


def get_expert_prediction(experts, prediction_set_i, method="voting"):
    ensemble_expert_pred_i = np.array(experts_test)[prediction_set_i][:, i]
    if method == "voting":
        exp_prediction = stats.mode(ensemble_expert_pred_i).mode if len(ensemble_expert_pred_i) != 0 else []

    if method == "last":
        exp_prediction = ensemble_expert_pred_i[-1] if len(ensemble_expert_pred_i) != 0 else []

    if method == "random":
        idx = np.random.randint(len(ensemble_expert_pred_i)) if len(ensemble_expert_pred_i) != 0 else -1
        exp_prediction = ensemble_expert_pred_i[idx] if idx != -1 else []

    return exp_prediction


# In[59]:


dict.fromkeys(["last", "random", "voting"])

# In[60]:


# Method dict ===
method_list = ["last", "random", "voting"]
method_dict = {"last": [],
               "random": [],
               "voting": []}

n_experts = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20]
alpha = 0.1
n_classes = 10

for method in method_list:

    print("Method: {}\n".format(method))
    for i, n in enumerate(n_experts):
        # =============
        # = Get Probs =
        # =============
        n_classes_exp = n_classes + n

        probs = confs[i]
        experts = exps[i]
        experts = experts[::-1]  # reverse order!
        y_true = true[-i]

        # Val/Calibration ===
        probs_val = probs[:n_val, n_classes:]
        experts_val = [exp[:n_val] for exp in experts]
        y_true_val = y_true[:n_val]

        # Test ===
        probs_test = probs[n_val:, n_classes:]
        experts_test = [exp[n_val:] for exp in experts]
        y_true_test = y_true[n_val:]

        # =============
        # = Conformal =
        # =============

        # Calculate Q_hat ===

        # === Only on deferred samples
        _, predicted = torch.max(probs[:n_val].data, 1)
        r = (predicted >= n_classes_exp - n)

        # Filter
        probs_experts = probs_val[r]
        experts_val = [np.array(exp)[r] for exp in experts_val]
        y_true_val = np.array(y_true_val)[r]

        # Model expert probs ===
        # Sort J model outputs for experts
        sort, pi = probs_experts.sort(dim=1, descending=True)

        # Correctness experts ===
        # Check if experts are correct
        correct_exp = (np.array(experts_val) == np.array(y_true_val)).T
        # idx for correct experts: [[0,1,2], [1,2], [], ...]
        correct_exp_idx = [np.where(correct_exp_i)[0] for correct_exp_i in correct_exp]

        # obtain the last expert to be retrieved. If empty, then add all values.
        # indexes are not the real expert index, but the sorted indexes, e.g. [[1, 0 ,2],  [1,0], [], ...]
        pi_corr_exp = [probs_experts[i, corr_exp].sort(descending=True)[1] for i, corr_exp in enumerate(correct_exp)]
        pi_corr_exp_stop = [pi_corr_exp_i[-1] if len(pi_corr_exp_i) != 0 else -1 for pi_corr_exp_i in
                            pi_corr_exp]  # last expert

        # obtain real expert index back, e.g. [2,1,-1,...]
        pi_stop = [correct_exp_idx[i][pi_corr_exp_stop_i] if len(correct_exp_idx[i]) != 0 else -1 for
                   i, pi_corr_exp_stop_i in enumerate(pi_corr_exp_stop)]

        scores = sort.cumsum(dim=1).gather(1, pi.argsort(1))[range(len(torch.tensor(pi_stop))), torch.tensor(pi_stop)]
        n_quantile = r.sum()
        qhat = torch.quantile(scores, np.ceil((n_quantile + 1) * (1 - alpha)) / n_quantile, interpolation="higher")

        print("Q_hat {}: {}".format(n, qhat))

        # =============
        # = Metrics =
        # =============

        # === Initalize ====

        correct = 0
        correct_sys = 0
        exp = 0
        exp_total = 0
        total = 0
        real_total = 0
        alone_correct = 0

        # Individual Expert Accuracies === #
        expert_correct_dic = {k: 0 for k in range(len(experts_test))}
        expert_total_dic = {k: 0 for k in range(len(experts_test))}

        probs_test_exp = probs_test
        probs_test_model = probs[n_val:]

        # Predicted value
        _, predicted = torch.max(probs_test_model.data, 1)

        # Classifier alone prediction
        _, prediction = torch.max(probs_test_model.data[:, :(n_classes_exp - n)], 1)

        labels = y_true_test
        for i in range(0, n_test):
            r = (predicted[i].item() >= n_classes_exp - n)
            alone_correct += (prediction[i] == labels[i]).item()

            # Non-deferred
            if r == 0:
                total += 1
                correct += (predicted[i] == labels[i]).item()
                correct_sys += (predicted[i] == labels[i]).item()

            # Deferred
            if r == 1:
                # Conformal prediction ===
                # Sort J model outputs for experts. sorted probs and sorted indexes
                sort_i, pi_i = probs_test_exp[i].sort(descending=True)
                # Get last sorted index to be below Q_hat
                pi_stop_i = (sort_i.cumsum(dim=0) <= qhat).sum()

                # Prediction sets
                prediction_set_i = (pi_i[:(pi_stop_i)]).numpy()  # not allow empty sets

                # - Get expert prediction depending on method
                # ======
                exp_prediction = get_expert_prediction(experts_test, prediction_set_i, method=method)
                # ======

                # Deferral accuracy: No matter expert ===
                exp += (exp_prediction == labels[i])
                exp_total += 1
                # Individual Expert Accuracy ===
                # expert_correct_dic[deferred_exp] += (exp_prediction == labels[i].item())
                # expert_total_dic[deferred_exp] += 1
                #
                correct_sys += (exp_prediction == labels[i])

            real_total += 1

        #  ===  Coverage  === #
        cov = 100 * total / real_total

        #  === Individual Expert Accuracies === #
        expert_accuracies = {"expert_{}".format(str(k)): 100 * expert_correct_dic[k] / (expert_total_dic[k] + 0.0002)
                             for k
                             in range(len(experts_test))}

        # Add expert accuracies dict
        to_print = {"coverage": cov,
                    "system_accuracy": 100 * correct_sys / real_total,
                    "expert_accuracy": 100 * exp / (exp_total + 0.0002),
                    "classifier_accuracy": 100 * correct / (total + 0.0001),
                    "alone_classifier": 100 * alone_correct / real_total}
        print(to_print, flush=True)

        # Save to method dict ===
        method_dict[method].append(to_print)

# In[61]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc

# # === Latex Options === #
rc('font', family='serif')
rc('text', usetex=True)

# === Matplotlib Options === #
cm = plt.cm.get_cmap('tab10')
plot_args = {"linestyle": "-",
             "marker": "o",
             "markeredgecolor": "k",
             "markersize": 10,
             "linewidth": 8
             }
sns.set_context("talk", font_scale=1.3)
fig_size = (6, 6)

# In[62]:


sys_acc_last = np.array([method_d["system_accuracy"] for method_d in method_dict["last"]])
sys_acc_last

# In[64]:


n_experts = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20]

sys_acc_last = np.array([method_d["system_accuracy"] for method_d in method_dict["last"]])
sys_acc_random = np.array([method_d["system_accuracy"] for method_d in method_dict["random"]])
sys_acc_voting = np.array([method_d["system_accuracy"] for method_d in method_dict["voting"]])

f, ax = plt.subplots(1, 1, figsize=fig_size)
ax.plot(n_experts, sys_acc_last, label=r"Last", **plot_args)
ax.plot(n_experts, sys_acc_random, label=r"Random", **plot_args)
ax.plot(n_experts, sys_acc_voting, label=r"Voting", **plot_args)
plt.xticks(n_experts, n_experts)
plt.yticks(list(plt.yticks()[0])[::2])
plt.ylabel(r'System Acc. ($\%$)')
plt.xlabel(r'\# Experts')
plt.title(r"OvA CIFAR-10")
plt.legend(loc="best")
plt.grid()
f.set_tight_layout(True)
plt.legend()

# In[66]:


n_experts = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20]

exp_acc_last = np.array([method_d["expert_accuracy"] for method_d in method_dict["last"]])
exp_acc_random = np.array([method_d["expert_accuracy"] for method_d in method_dict["random"]])
exp_acc_voting = np.array([method_d["expert_accuracy"] for method_d in method_dict["voting"]])

f, ax = plt.subplots(1, 1, figsize=fig_size)
ax.plot(n_experts, exp_acc_last, label=r"Last", **plot_args)
ax.plot(n_experts, exp_acc_random, label=r"Random", **plot_args)
ax.plot(n_experts, exp_acc_voting, label=r"Voting", **plot_args)
plt.xticks(n_experts, n_experts)
plt.yticks(list(plt.yticks()[0])[::2])
plt.ylabel(r'Exp Acc. ($\%$)')
plt.xlabel(r'\# Experts')
plt.title(r"OvA CIFAR-10")
plt.legend(loc="best")
plt.grid()
f.set_tight_layout(True)
plt.legend()

# In[ ]:


n_experts = [1, 2, 4, 6, 8, 10, 12, 16, 18, 20]

coverage = np.array([method_d["coverage"] for method_d in method_dict["last"]])

f, ax = plt.subplots(1, 1, figsize=fig_size)
ax.plot(n_experts, coverage, label=r"Coverage", **plot_args)
plt.xticks(n_experts, n_experts)
plt.yticks(list(plt.yticks()[0])[::2])
plt.ylabel(r'Model Coverage. ($\%$)')
plt.xlabel(r'\# Experts')
plt.title(r"OvA CIFAR-10")
plt.legend(loc="best")
plt.grid()
f.set_tight_layout(True)
plt.legend()

