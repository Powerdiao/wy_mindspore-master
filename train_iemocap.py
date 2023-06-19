from __future__ import print_function
from model_mindspore import LMF
from utils_mindspore import total, load_iemocap
from mindspore import Parameter
from mindspore import ops
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import os
import argparse
import mindspore as ms
import random
import mindspore.nn as nn
import numpy as np
import csv
 

def display(f1_score, accuracy_score):
    print("F1-score on test set is {}".format(f1_score))
    print("Accuracy score on test set is {}".format(accuracy_score))


def main(options):
    DTYPE = ms.float32
    LONG = ms.int64

    # parse the input args
    epochs = options['epochs']
    data_path = options['data_path']
    model_path = options['model_path']
    output_path = options['output_path']
    signiture = options['signiture']
    patience = options['patience']
    emotion = options['emotion'].encode()
    output_dim = options['output_dim']

    # prepare the paths for storing models and outputs
    model_path = os.path.join(
        model_path, "model_{}_{}.ckpt".format(signiture, emotion))
    output_path = os.path.join(
        output_path, "results_{}_{}.csv".format(signiture, emotion))
    print("Temp location for models: {}".format(model_path))
    print("Grid search results are in: {}".format(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    

    params = dict()
    params['audio_hidden'] = [options['audio_hidden']]
    params['video_hidden'] = [options['video_hidden']]
    params['text_hidden'] = [options['text_hidden']]
    params['audio_dropout'] = [options['audio_dropout']]
    params['video_dropout'] = [options['video_dropout']]
    params['text_dropout'] = [options['text_dropout']]
    params['factor_learning_rate'] = [options['factor_learning_rate']]
    params['learning_rate'] = [options['learning_rate']]
    params['rank'] = [options['rank']]
    params['batch_size'] = [options['batch_size']]
    params['weight_decay'] = [options['weight_decay']]

    total_settings = total(params)

    print("There are {} different hyper-parameter settings in total.".format(total_settings))

    seen_settings = set()

    if not os.path.isfile(output_path):
        with open(output_path, 'w+') as out:
            writer = csv.writer(out)
            writer.writerow(
                ["audio_hidden", "video_hidden", 'text_hidden', 'audio_dropout', 'video_dropout', 'text_dropout',
                 'factor_learning_rate', 'learning_rate', 'rank', 'batch_size', 'weight_decay',
                 'Best Validation CrossEntropyLoss', 'Test CrossEntropyLoss', 'Test F1-score', 'Test Accuracy Score'])

    
    for i in range(total_settings):
        ahid = random.choice(params['audio_hidden'])
        vhid = random.choice(params['video_hidden'])
        thid = random.choice(params['text_hidden'])
        thid_2 = thid // 2
        adr = random.choice(params['audio_dropout'])
        vdr = random.choice(params['video_dropout'])
        tdr = random.choice(params['text_dropout'])
        factor_lr = random.choice(params['factor_learning_rate'])
        lr = random.choice(params['learning_rate'])
        r = random.choice(params['rank'])
        batch_sz = random.choice(params['batch_size'])
        decay = random.choice(params['weight_decay'])

        # reject the setting if it has been tried
        current_setting = (ahid, vhid, thid, adr, vdr, tdr, factor_lr, lr, r, batch_sz, decay)
        # if current_setting in seen_settings:
            # continue
        # else:
            # seen_settings.add(current_setting)

        train_set, valid_set, test_set, input_dims = load_iemocap(data_path, emotion)

        model = LMF(input_dims, (ahid, vhid, thid), thid_2, (adr, vdr, tdr, 0.5), output_dim, r)
        print("Model initialized")
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        factors = list(model.get_parameters())[:3]
        other = list(model.get_parameters())[3:]
        optimizer = nn.Adam([{"params": factors, "lr": factor_lr}, {"params": other, "lr": lr}], weight_decay=decay)

        # setup training
        complete = True
        min_valid_loss = float('Inf')
        train_iterator = train_set.batch(batch_sz)
        valid_iterator = valid_set.batch(valid_set.get_dataset_size())
        test_iterator = test_set.batch(test_set.get_dataset_size())
        curr_patience = patience

        def forward_fn(data, label):
            logits = model(data[0], data[1], data[2])
            argmax = ops.ArgMaxWithValue(axis=1)
            index, _ = argmax(label)
            loss = loss_fn(logits, index)
            return loss, logits

        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            loss = ops.depend(loss, optimizer(grads))
            return loss

        for e in range(epochs):
            model.set_train()
            avg_train_loss = 0.0
            for batch in train_iterator.create_tuple_iterator(output_numpy=True):
                x = batch[:-1]
                x_a = Parameter(ms.Tensor(x[0], DTYPE), requires_grad=False)
                x_v = Parameter(ms.Tensor(x[1], DTYPE), requires_grad=False)
                x_t = Parameter(ms.Tensor(x[2], DTYPE), requires_grad=False)
                y = batch[-1].reshape(-1, output_dim)
                y = Parameter(ms.Tensor(y, LONG), requires_grad=False)
                avg_loss = train_step((x_a, x_v, x_t), y)
                avg_train_loss += avg_loss / train_set.get_dataset_size()

            print("Epoch {} complete! Average Training loss: {}".format(e, avg_train_loss))

            # Terminate the training process if run into NaN
            if avg_train_loss != avg_train_loss:
                print("Training got into NaN values...\n\n")
                complete = False
                break

            model.set_train(False)
            for batch in valid_iterator.create_tuple_iterator(output_numpy=True):
                x = batch[:-1]
                x_a = Parameter(ms.Tensor(x[0], DTYPE), requires_grad=False).squeeze()
                x_v = Parameter(ms.Tensor(x[1], DTYPE), requires_grad=False).squeeze()
                x_t = Parameter(ms.Tensor(x[2], DTYPE), requires_grad=False)
                y = batch[-1].reshape(-1, output_dim)
                y = Parameter(ms.Tensor(y, LONG), requires_grad=False)
                output = model(x_a, x_v, x_t)
                argmax = ops.ArgMaxWithValue(axis=1)
                index, _ = argmax(y)
                valid_loss = loss_fn(output, index)
                avg_valid_loss = valid_loss
            y = y.asnumpy().reshape(-1, output_dim)

            if avg_valid_loss != avg_valid_loss:
                print("Training got into NaN values...\n\n")
                complete = False
                break

            avg_valid_loss = avg_valid_loss / valid_set.get_dataset_size()
            print("Validation loss is: {}".format(avg_valid_loss))

            if avg_valid_loss < min_valid_loss:
                curr_patience = patience
                min_valid_loss = avg_valid_loss
                ms.save_checkpoint(model, model_path)
                print("Found new best model, saving to disk...")
            else:
                curr_patience -= 1

            if curr_patience <= 0:
                break
            print("\n\n")

        if complete:
            best_params = ms.load_checkpoint(model_path)
            ms.load_param_into_net(model, best_params)
            model.set_train(False)
            for batch in test_iterator.create_tuple_iterator(output_numpy=True):
                x = batch[:-1]
                x_a = Parameter(ms.Tensor(x[0], DTYPE), requires_grad=False).squeeze()
                x_v = Parameter(ms.Tensor(x[1], DTYPE), requires_grad=False).squeeze()
                x_t = Parameter(ms.Tensor(x[2], DTYPE), requires_grad=False)
                y = batch[-1].reshape(-1, output_dim)
                y = Parameter(ms.Tensor(y, LONG), requires_grad=False)
                output_test = model(x_a, x_v, x_t)
                argmax = ops.ArgMaxWithValue(axis=1)
                index, _ = argmax(y)
                loss_test = loss_fn(output_test, index)
                # test_loss = loss_test.data[0]
                test_loss = loss_test

            output_test = output_test.asnumpy().reshape(-1, output_dim)
            y = y.asnumpy().reshape(-1, output_dim)
            test_loss = test_loss / test_set.get_dataset_size()

            # these are the needed metrics
            all_true_label = np.argmax(y, axis=1)
            all_predicted_label = np.argmax(output_test, axis=1)

            f1 = f1_score(all_true_label, all_predicted_label, average='weighted')
            acc_score = accuracy_score(all_true_label, all_predicted_label)

            display(f1, acc_score)

            with open(output_path, 'a+') as out:
                writer = csv.writer(out)
                writer.writerow([ahid, vhid, thid, adr, vdr, tdr, factor_lr, lr, r, batch_sz, decay,
                                 min_valid_loss, test_loss, f1, acc_score])


if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--emotion', dest='emotion', type=str, default=b'sad')
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=100)
    OPTIONS.add_argument('--output_dim', dest='output_dim', type=int, default=2)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--signiture', dest='signiture', type=str, default='')
    OPTIONS.add_argument('--cuda', dest='cuda', type=bool, default=False)
    OPTIONS.add_argument('--data_path', dest='data_path',
                         type=str, default='./data/')
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='models')
    OPTIONS.add_argument('--output_path', dest='output_path',
                         type=str, default='results/mindspore')
    OPTIONS.add_argument('--audio_hidden', dest='audio_hidden',
                          type=int, default=4)
    OPTIONS.add_argument('--video_hidden', dest='video_hidden',
                          type=int, default=4)
    OPTIONS.add_argument('--text_hidden', dest='text_hidden',
                          type=int, default=4)                      
    OPTIONS.add_argument('--audio_dropout', dest='audio_dropout',
                          type=float, default=0.3)                
    OPTIONS.add_argument('--video_dropout', dest='video_dropout',
                          type=float, default=0.5)
    OPTIONS.add_argument('--text_dropout', dest='text_dropout',
                          type=float, default=0.2)
    OPTIONS.add_argument('--factor_learning_rate', dest='factor_learning_rate',
                          type=float, default=0.0005) 
    OPTIONS.add_argument('--learning_rate', dest='learning_rate',
                          type=float, default=0.001) 
    OPTIONS.add_argument('--rank', dest='rank',
                          type=int, default=1)
    OPTIONS.add_argument('--batch_size', dest='batch_size',
                          type=int, default=64) 
    OPTIONS.add_argument('--weight_decay', dest='weight_decay',
                          type=float, default=0)                     
    PARAMS = vars(OPTIONS.parse_args())
    main(PARAMS)
