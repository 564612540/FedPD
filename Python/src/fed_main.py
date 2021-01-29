

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
# from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNFEMnist
from utils import get_dataset, average_weights, exp_details
from agent import Agent

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    # logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'femnist':
            global_model = CNNFEMnist(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    # global_model.to(device)
    global_model.train()
    # print(global_model)

    agent_list = []
    for i in range(args.num_users):
        agent_list.append(Agent(global_model,args,i, nn.NLLLoss().to(device)))
    
    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 5
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights = []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        m = args.num_users
        if epoch % args.freq_out ==0:
            compute_full = True
            update_model = True
        else:
            compute_full = False
            update_model = False
        global_model.train()
        for idx in range(args.num_users):
            w= agent_list[idx].train_(global_model, args.freq_in, train_dataset, user_groups, update_model, compute_full)
            # print(w['layer3.0.bias'])
            local_weights.append(copy.deepcopy(w))
        # for idx in range(args.num_users):
        #     print(local_weights[idx]['layer3.0.bias'])

        w_avg = copy.deepcopy(local_weights[0])
        for key in w_avg.keys():
            for i in range(1, args.num_users):
                w_avg[key].add_(local_weights[i][key])
            w_avg[key].div_(args.num_users)
        # print(w_avg['layer3.0.bias'])

        # update global weights
        global_weights = w_avg

        # update global weights
        global_model.load_state_dict(global_weights)
        # print(global_model)

        # Calculate avg training accuracy over all users at every epoch
        global_model.eval()
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        train_accuracy.append(test_acc)
        train_loss.append(test_loss)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]),flush=True)

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_T[{}]_Q[{}]_I[{}]_B[{}]_lr[{}].csv'.\
        format(args.dataset, args.optimizer, args.epochs,
               args.local_ep, args.freq_out, args.local_bs, args.lr)

    with open(file_name, 'w') as f:
        # pickle.dump([train_loss, train_accuracy], f)
        f.write("\n".join(str([loss, acc]) for loss, acc in zip(train_loss, train_accuracy)))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
