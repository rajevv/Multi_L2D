import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import os
import pickle5 as pickle
import torchvision.models as models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,  flush=True)

def load_data(file_name):
    assert(os.path.exists(file_name+'.pkl'))
    with open(file_name + '.pkl', 'rb') as f:
        data = pickle.load(f)
    return data



def save_data(data, file_path):
    with open(file_path + '.pkl','wb') as f:
        pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)


# #### The Galaxy-zoo data can be found [here](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge).  Data is prepared in "prepare_data.py" and saved in "galaxy_data.pkl". Refer to section 6 of the paper for a detailed description of preprocessing and human predictions modeling. In this notebook we only load the data which we previously generated using the "prepare_data.py".


constraints = [0.0,0.2,0.4,0.6,0.8,1.0]
data_path = 'galaxy_data'
model_dir = 'models/'
res_dir = 'results/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(res_dir):
    os.mkdir(res_dir)

def metrics_print_ova(net, val_X, val_Y, val_human_is_correct):
    net.eval()
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0
    batch_size = 64
    num_batches = int(val_X.shape[0] / batch_size)
    n_classes = 2
    with torch.no_grad():
        for i in range(num_batches):
            X_batch = val_X[i * batch_size:(i + 1) * batch_size]
            Y_batch = val_Y[i * batch_size:(i + 1) * batch_size]
            val_human_is_correct_batch = val_human_is_correct[i * batch_size:(i + 1) * batch_size]
            outputs = F.sigmoid(net(X_batch))
            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]            # batch_size
            for i in range(0,batch_size):
                r = (predicted[i].item() == n_classes)
                prediction = predicted[i]
                if predicted[i] == n_classes:
                    max_idx = 0
					          # get second max
                    for j in range(0, n_classes):
                        if outputs.data[i][j] >= outputs.data[i][max_idx]:
                          max_idx = j
                    prediction = max_idx
                else:
                  prediction = predicted[i]
                alone_correct += (prediction == Y_batch[i]).item()
                if r==0:
                    total += 1
                    correct += (predicted[i] == Y_batch[i]).item()
                    correct_sys += (predicted[i] == Y_batch[i]).item()
                if r==1:
                    exp += val_human_is_correct_batch[i].item()
                    correct_sys += val_human_is_correct_batch[i].item()
                    exp_total+=1
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print={"coverage":cov, "system accuracy": 100*correct_sys/real_total, "expert accuracy":100* exp/(exp_total+0.0002),"classifier accuracy":100*correct/(total+0.0001), "alone classifier": 100*alone_correct/real_total }
    print(to_print)
    return [100*total/real_total,  100*correct_sys/real_total, 100* exp/(exp_total+0.0002),100*correct/(total+0.0001) ]

def loss_func_ova(outputs, labels, m, n_classes):
	batch_size = outputs.size()[0]
	l1 = LogisticLoss(outputs[range(batch_size), labels], 1)
	l2 = torch.sum(LogisticLoss(outputs[:,:n_classes], -1), dim=1) - LogisticLoss(outputs[range(batch_size),labels],-1)
	l3 = LogisticLoss(outputs[range(batch_size), n_classes], -1)

	l4 = LogisticLoss(outputs[range(batch_size), n_classes], 1)

	l5 = m*(l4 - l3)

	l = l1 + l2 + l3 + l5
	return torch.mean(l)


def LogisticLoss(outputs, y):
	outputs[torch.where(outputs==0.0)] = (-1*y)*(-1*np.inf)
	l = torch.log2(1 + torch.exp((-1*y)*outputs))
	return l

def surrogate_train_ova(data_path):  
    print('-----training machine model : surrogate_ova',  flush=True)
    machine_type = 'surrogate_ova'
    data = load_data(data_path)
    X = torch.from_numpy(data['X']).float().to(device)
    Y = torch.from_numpy(data['Y']).to(device).long()
    human_is_correct = torch.from_numpy(np.array([1 if data['hpred'][i]==data['Y'][i] else 0
                                                  for i in range(X.shape[0])])).to(device)
    alpha = 1.0
    m = human_is_correct * 1.0
    m.to(device)


    val_X = torch.from_numpy(data['val']['X']).float().to(device)
    val_Y = torch.from_numpy(data['val']['Y']).to(device).long()
    val_human_is_correct = torch.from_numpy(np.array([1 if data['val']['hpred'][i] == data['val']['Y'][i] else 0
                                                      for i in range(val_X.shape[0])])).to(device)
    val_m = val_human_is_correct * 1.0
    val_m.to(device)
    
    batch_size = 128
    num_epochs = 200

    num_batches = int(X.shape[0] / batch_size)
    
    val_batches = int(val_X.shape[0] / batch_size)
    
    output_dim = 2
    mnet = models.resnet50()
    mnet.fc = torch.nn.Sequential(
        nn.Linear(2048, output_dim + 1)
    )
    mnet.to(device)

    lrate = 0.0001
    warmup_iters = 5*num_batches
    optimizer = torch.optim.Adam(mnet.parameters(),lr=lrate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_batches * num_epochs)
    train_losses = []
    val_losses = []
    best_val_loss = 1000
    max_patience = 20
    patience = 0
    eps = 1e-4
    iter = 0
    for epoch in range(num_epochs):
        print('----- epoch:',epoch, '-----',  flush=True)
        epoch_loss = 0
        val_epoch_loss = 0
        for i in range(num_batches):

            ## for warmp-ups to avoid overfitting
            if iter < warmup_iters:
                lr = lrate*float(iter) / warmup_iters
                print(iter, lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            X_batch = X[i * batch_size:(i + 1) * batch_size]
            Y_batch = Y[i * batch_size:(i + 1) * batch_size]
            m_batch = m[i * batch_size:(i + 1) * batch_size]

            optimizer.zero_grad()
            loss = loss_func_ova(mnet(X_batch),Y_batch,m_batch,output_dim)
            loss.backward()
            optimizer.step()
            ## initiate scheduler only after warmup period
            if not iter < warmup_iters:
                scheduler.step()
            epoch_loss += float(loss)
            iter+=1


        print('train loss: ',epoch_loss/ num_batches,  flush=True)
        train_losses.append(epoch_loss / num_batches)
        _ = metrics_print_ova(mnet, val_X, val_Y, val_human_is_correct)
        with torch.no_grad():
            vloss = []
            for i in range(val_batches):
                val_X_batch = val_X[i * batch_size:(i + 1) * batch_size]
                val_Y_batch = val_Y[i * batch_size:(i + 1) * batch_size]
                val_m_batch = val_m[i * batch_size:(i + 1) * batch_size]
                m_val_prob = mnet(val_X_batch)
                val_loss = loss_func_ova(m_val_prob, val_Y_batch, val_m_batch, output_dim)
                vloss.append(val_loss.item())
            
            val_loss = np.average(vloss)
            val_losses.append(val_loss)
            print('validation loss: ', float(val_loss), flush=True)
            
            if val_loss < best_val_loss:
                torch.save(mnet.state_dict(), model_dir + 'm_surrogate_ova')
                best_val_loss = val_loss
                print('updated the model', flush=True)
                patience = 0
            else:
                patience += 1
            
            if patience > max_patience:
                print('no progress for 10 epochs... stopping training', flush=True)
                break
    
        print('\n')


surrogate_train_ova(data_path)




data = load_data(data_path)
test_X = torch.from_numpy(data['test']['X']).float().to(device)
test_Y = data['test']['Y']
hlabel = data['test']['hpred']
test_human_is_correct = torch.from_numpy(np.array([1 if data['test']['hpred'][i] == data['test']['Y'][i] else 0
                                                      for i in range(test_X.shape[0])])).to(device)
mnet = models.resnet50()
mnet.fc = torch.nn.Sequential(
            nn.Linear(2048, 2 + 1)
        )
mnet.to(device)

mnet.load_state_dict(torch.load(model_dir + 'm_surrogate_ova'))
mnet.eval()


metrics_print_ova(mnet, test_X, test_Y, test_human_is_correct)
