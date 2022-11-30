import numpy.random as rand
import codecs
import csv
import random
import numpy as np
import numpy.linalg as LA
import fasttext
import copy
import pickle5 as pickle
import os
import torch

'''
The data can be downloaded from https://github.com/t-davidson/hate-speech-and-offensive-language. put the labeled_data.csv in this directory and run this script using python3 process_text_data.py
'''


def load_data(file_name):
    assert(os.path.exists(file_name+'.pkl'))
    with open(file_name + '.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def save_data(data, file_path):
    with open(file_path + '.pkl','wb') as f:
        pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)  

class preprocess_triage_real_data:
    def __init__(self):
        pass

    def process_hate_speech_data(self, src_file, dest_file):

        with open(src_file, 'r') as f:
            f.readline()
            dict_tweet = {}
            response_list = []
            human_annotation_list = []
            while True:
                line_full = f.readline()
                if not line_full:
                    save_data({'tweets': dict_tweet, 'y': response_list, 'y_h': human_annotation_list}, dest_file)
                    return
                else:
                    if line_full.isspace():
                        print('empty')
                    else:
                        line = line_full.split(',', 7)
                        if len(line) == 7:
                            tid = line[0]
                            tweet = line[-1]
                            dict_tweet[tid] = tweet
                            y, y_h = self.get_annotations(line[1:-1])
                            response_list.append(y)
                            human_annotation_list.append(y_h)

    def get_annotations(self, list_of_arg):
        human_response = []
        for i in [1, 2, 3]:
            if int(list_of_arg[i]) > 0:
                human_response.extend([i - 1] * int(list_of_arg[i]))
        response = int(list_of_arg[-1])
        return response, human_response

    def dict_to_txt(self, tweet_dict, file_w):
        with open(file_w, 'w') as f:
            for tweet in tweet_dict.values():
                f.write(tweet)

    def map_range(self, v, l, h, l_new, h_new):
        return float(v - l) * ((h_new - l_new) / float(h - l)) + l_new

    def convert_tweet_to_vector(self, file_dict, file_vec, file_tweet):
        epsilon = 0.01
        data_dict = load_data(file_dict)
        data_vec = {}
        n_data = len(data_dict['y'])
        print(n_data)
        data_vec['y'] = np.array(data_dict['y'])
        data_vec['c'] = np.zeros((n_data,3))
        data_vec['hpred'] = np.zeros(n_data)

        for ind, human_pred, response in zip(range(n_data), data_dict['y_h'], data_vec['y']):

            h0 = float(np.sum([hpred==0 for hpred in human_pred]))
            h1 = float(np.sum([hpred==1 for hpred in human_pred]))
            h2 = float(np.sum([hpred==2 for hpred in human_pred]))
            assert(h0 + h1 + h2 == len(human_pred))
            total_votes = float(len(human_pred))
            data_vec['c'][ind] = np.array([float(h0/total_votes),float(h1/total_votes),float(h2/total_votes)])
            for i,val in enumerate(data_vec['c'][ind]):
                if val<epsilon:
                    data_vec['c'][ind][i] = epsilon
                    data_vec['c'][ind][np.argmax(data_vec['c'][ind])] -= epsilon

            human = np.random.choice(len(human_pred))
            data_vec['hpred'][ind] = human_pred[human]

        self.dict_to_txt(data_dict['tweets'],file_tweet)
        model = fasttext.train_unsupervised(file_tweet, model='skipgram')
        x = []
        for tid in data_dict['tweets'].keys():
            tweet = data_dict['tweets'][tid].replace('\n', ' ')
            x.append(model.get_sentence_vector(tweet).flatten())
        data_vec['x'] = np.array(x)

        save_data(data_vec, file_vec)

    def truncate_data(self, data_file, data_file_tr):
        data = load_data(data_file)
        n = data['y'].shape[0]
        n_tr = int(n / 4)
        print('x', data['x'].shape)
        print('y', data['y'].shape)
        print('c', data['c'].shape)
        data['x'] = data['x'][:n_tr]
        data['y'] = data['y'][:n_tr]
        data['c'] = data['c'][:n_tr]
        data['hpred'] = data['hpred'][:n_tr]
        print('x', data['x'].shape)
        print('y', data['y'].shape)
        print('c', data['c'].shape)
        save_data(data, data_file_tr)

    def split_data(self, frac, file_data, file_data_split):

        data = load_data(file_data)

        print('x', data['x'].shape)
        print('y', data['y'].shape)
        print('c', data['c'].shape)
        num_data = data['y'].shape[0]
        print(num_data)
        num_train = int(frac * num_data)
        num_test = int((num_data - num_train)/2)
        num_val = num_data - (num_test + num_train)
        indices = np.arange(num_data)
        random.shuffle(indices)
        indices_train = indices[:num_train]
        indices_val = indices[num_train:num_train+num_val]
        indices_test = indices[num_train+num_val:num_train+num_val+num_test]
        data_split = {}
        data_split['X'] = data['x'][indices_train]
        data_split['Y'] = data['y'][indices_train]
        data_split['c'] = data['c'][indices_train]
        data_split['hpred'] = data['hpred'][indices_train]

        val = {}
        val['X'] = data['x'][indices_val]
        val['Y'] = data['y'][indices_val]
        val['c'] = data['c'][indices_val]
        val['hpred'] = data['hpred'][indices_val]
        data_split['val'] = val

        test = {}
        test['X'] = data['x'][indices_test]
        test['Y'] = data['y'][indices_test]
        test['c'] = data['c'][indices_test]
        test['hpred'] = data['hpred'][indices_test]
        data_split['test'] = test
        data_split['dist_mat'] = np.zeros((num_test, num_train))
        save_data(data_split, file_data_split)

    def change_format_hatespeech(self, data_file, dest_file):
        data = load_data(data_file)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        x_train = scaler.fit_transform(data['X'])
        x_val = scaler.transform(data['val']['X'])
        x_test = scaler.transform(data['test']['X'])
        data['X'] = x_train
        data['val']['X'] = x_val
        data['test']['X'] = x_test

        c = {'0.0': np.copy(data['c'])}
        val_c = {'0.0': np.copy(data['val']['c'])}
        test_c = {'0.0': np.copy(data['test']['c'])}
        data['c'] = c
        data['val']['c'] = val_c
        data['test']['c'] = test_c
        save_data(data, dest_file)

def find_human_loss(file_path):
    data = load_data(file_path)
    loss_func = torch.nn.NLLLoss(reduction='none')

    Y = torch.from_numpy(copy.deepcopy(data['Y'])).long()
    hprob = torch.log2(torch.from_numpy(copy.deepcopy(data['c']['0.0'])).float())
    hloss = loss_func(hprob, Y)
    human_prob, _ = torch.max(hprob, axis=1)

    val_Y = torch.from_numpy(copy.deepcopy(data['val']['Y'])).long()
    val_hprob = torch.log2(torch.from_numpy(copy.deepcopy(data['val']['c']['0.0'])).float())
    val_hloss = loss_func(val_hprob, val_Y)
    val_human_prob, _ = torch.max(val_hprob, axis=1)


    test_Y = torch.from_numpy(copy.deepcopy(data['test']['Y'])).long()
    test_hprob = torch.log2(torch.from_numpy(copy.deepcopy(data['test']['c']['0.0'])).float())
    test_hloss = loss_func(test_hprob, test_Y)
    test_human_prob, _ = torch.max(test_hprob, axis=1)

    data['hloss'] = hloss
    data['val']['hloss'] = val_hloss
    data['test']['hloss'] = test_hloss

    data['hprob'] = human_prob
    data['val']['hprob'] = val_human_prob
    data['test']['hprob'] = test_human_prob
    save_data(data,file_path)

def main():
    path = ''
    src_file = path + 'labeled_data.csv'
    obj = preprocess_triage_real_data()
    dest_file = path + 'data'
    tweet_file = path + 'tweets.txt'
    vec_file = path + 'data_vectorized'
    vec_full_split_file = path + 'input_full'

    obj.process_hate_speech_data(src_file,dest_file)
    obj.convert_tweet_to_vector(dest_file,vec_file,tweet_file)
    obj.split_data(0.6, vec_file , vec_full_split_file)

    dest_file = 'hatespeech_data'
    obj.change_format_hatespeech(vec_full_split_file, dest_file)
    find_human_loss(dest_file)


if __name__ == "__main__":
    main()
