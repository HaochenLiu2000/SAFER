from models import *
import wandb
import os
import sys
import torch
import shutil
import logging
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import transformers
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pdb
import numpy as np
from load_kg_dataset import PairSubgraphsFewShotDataLoader
import torch.nn.functional as F
import optuna

class Trainer:
    def __init__(self, data_loaders, parameter):
        self.parameter = parameter
        self.train_data_loader=data_loaders
        # parameters
        self.few = parameter['few']
        self.num_query = parameter['num_query']
        self.batch_size = parameter['batch_size']
        self.learning_rate = parameter['learning_rate']
        self.early_stopping_patience = parameter['early_stopping_patience']
        self.niters = parameter['niters']
        self.threshold = parameter['threshold']
        # epoch
        self.epoch = parameter['epoch']
        self.print_epoch = parameter['print_epoch']
        self.eval_epoch = parameter['eval_epoch']
        self.checkpoint_epoch = parameter['checkpoint_epoch']
        # device
        self.device = parameter['device']
        self.margin = parameter['margin']
        self.debug = parameter['debug']
        self.pretrain_on_bg = parameter['pretrain_on_bg']

        self.parameter['prefix'] = self.parameter['prefix'] + "_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        
        #self.model1_ex=matching_model1(self.train_data_loader.dataset, parameter)
        self.model2_ex=mix_model2(self.train_data_loader.dataset, parameter)
        
        if self.parameter['step'] == "pretrain2":
            self.optimizer_model2 = optim.AdamW(filter(lambda p: p.requires_grad, self.model2_ex.parameters()), lr=self.learning_rate)
            self.scheduler_model2 = transformers.get_linear_schedule_with_warmup(self.optimizer_model2, 2000, self.epoch)
            #self.scheduler_model2 = transformers.get_constant_schedule_with_warmup(self.optimizer_model2, 1000)
            


            
    def rank_predict(self, data, x, ranks):
        # query_idx is the idx of positive score
        query_idx = x.shape[0] - 1
        # sort all scores with descending, because more plausible triple has higher score
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        # update data
        if rank <= 10:
            data['Hits@10'] += 1
        if rank <= 5:
            data['Hits@5'] += 1
        if rank == 1:
            data['Hits@1'] += 1
        data['MRR'] += 1.0 / rank
        return rank

    #special case study to read the relaitons
    def case(self,eval_task):
        support, support_subgraphs, support_negative, support_negative_subgraphs, query, query_subgraphs, negative, negative_subgraphs = eval_task
        raw_data_paths=os.path.join(self.parameter["data_path"], self.parameter['dataset'])
        with open(os.path.join(raw_data_paths, f'relation2id.json'), 'r') as f:
            relation2id = json.load(f)  
        with open(os.path.join(raw_data_paths, f'entity2id.json'), 'r') as f:
            entity2id = json.load(f) 
        id2relation = {v: k for k, v in relation2id.items()}
        id2entity = {v: k for k, v in entity2id.items()}
        s_subgraphs=Batch.from_data_list(support_subgraphs).to(self.device)
        q_subgraphs=Batch.from_data_list(support_subgraphs+query_subgraphs).to(self.device)
        batch=q_subgraphs.batch
        batch_num_nodes = scatter_sum(torch.ones(batch.shape).to(self.device), batch)     
        head_idxs_support = torch.cumsum(torch.cat([torch.tensor([0]).to(self.device),batch_num_nodes[:-1]]), 0).long()
        tail_idxs_support = torch.cumsum(torch.cat([torch.tensor([0]).to(self.device),batch_num_nodes[:-1]]), 0).long() + 1
        
        
        _, masksgt1, _ = self.metaR.sample_connected_masks(s_subgraphs, kk=1)
        #exit()
        _, masksgt2, _ = self.metaR.sample_connected_masks(q_subgraphs, kk=1)
        #print(masksgt1,masksgt1.shape)
        #print(masksgt2,masksgt2.shape)
        print(support)
        print(query)
        #for i in range(len(s_subgraphs.edge_attr)):
        #    if s_subgraphs.edge_index[0][i]<head_idxs_support[1]:
        #        print('0   ',id2entity[int(s_subgraphs.x_id[s_subgraphs.edge_index[0][i]])],'   ',id2relation[int(s_subgraphs.edge_attr[i])],'   ',id2entity[int(s_subgraphs.x_id[s_subgraphs.edge_index[1][i]])])
        #    else:
        #        print('1   ',id2entity[int(s_subgraphs.x_id[s_subgraphs.edge_index[0][i]])],'   ',id2relation[int(s_subgraphs.edge_attr[i])],'   ',id2entity[int(s_subgraphs.x_id[s_subgraphs.edge_index[1][i]])])
        mask1=torch.nonzero(masksgt1==1).squeeze(-1)
        #print(mask1,mask1.shape)
        l1=len(mask1)
        for i in range(l1):
            if s_subgraphs.edge_index[0][mask1[i]]<head_idxs_support[1]:
                print('0c  ',id2entity[int(s_subgraphs.x_id[s_subgraphs.edge_index[0][mask1[i]]])],'   ',id2relation[int(s_subgraphs.edge_attr[mask1[i]])],'   ',id2entity[int(s_subgraphs.x_id[s_subgraphs.edge_index[1][mask1[i]]])])
            else:
                print('1c  ',id2entity[int(s_subgraphs.x_id[s_subgraphs.edge_index[0][mask1[i]]])],'   ',id2relation[int(s_subgraphs.edge_attr[mask1[i]])],'   ',id2entity[int(s_subgraphs.x_id[s_subgraphs.edge_index[1][mask1[i]]])])
        mask2=torch.nonzero(masksgt2==1).squeeze(-1)
        #print(mask2,mask2.shape)
        l2=len(mask2)
        for i in range(l2):
            if q_subgraphs.edge_index[0][mask2[i]]>=head_idxs_support[2]:
                print('qc  ',id2entity[int(q_subgraphs.x_id[q_subgraphs.edge_index[0][mask2[i]]])],'   ',id2relation[int(q_subgraphs.edge_attr[mask2[i]])],'   ',id2entity[int(q_subgraphs.x_id[q_subgraphs.edge_index[1][mask2[i]]])])
        
        exit()  
        
    def model2(self, test_data_loader_ranktail4, eadd1=1,istest=False, epoch=None, trial = None, best_params = None):
        #testing
        #self.model2_ex.eval()
        if istest:
            data_loader = test_data_loader_ranktail4
        else:
            data_loader = test_data_loader_ranktail4
        
        if self.parameter['prev_state_dir_model2'] is not None:
            prev_ckpt = torch.load(self.parameter['prev_state_dir_model2'], map_location='cpu')
            self.model2_ex.load_state_dict(prev_ckpt, strict=False)
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        data2 = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        ranks = []

        t = 0
        temp = dict()
       
        for batch_idx, batch in tqdm(enumerate(data_loader), total = len(data_loader)):
            # sample all the eval tasks
            #print("batch_idx",batch_idx,"batch",batch)
            #exit(0)
            eval_task, curr_rel = batch
            # at the end of sample tasks, a symbol 'EOT' will return
            #print(batch)
            #self.case(eval_task)
            
            query_ans_list, negative_ans_list, query_sim_train, negative_sim_train = self.model2_ex(eval_task, iseval=False, is_eval_loss = True , curr_rel=curr_rel, trial = trial, best_params = best_params)
            #query_ans_list_for_supports_sim, negative_ans_list_for_supports_sim, query_ans_list_for_supports_cla, negative_ans_list_for_supports_cla, query_sim_train, negative_sim_train, query_cla_train, negative_cla_train = self.model2_ex(eval_task, iseval=False, is_eval_loss = True , curr_rel=curr_rel, trial = trial, best_params = best_params)
            
            x = torch.cat([negative_ans_list.reshape(len(curr_rel), -1), query_ans_list.reshape(1, -1)], 1)
            for idx in range(x.shape[0]):
                t += 1
                rank=self.rank_predict(data, x[idx], ranks)
                
        # print overall evaluation result and return it
        for k in data.keys():
            data[k] = round(data[k] / t, 3)
    
        print("{}\t{}\tMRR: {:.3f}\tHits@1: {:.3f}\tHits@5: {:.3f}\tHits@10: {:.3f}\r".format(
               t, eadd1, data['MRR'], data['Hits@1'], data['Hits@5'], data['Hits@10']))
        return data['MRR']
    
    
    def pretrain2(self, pretrain2_data_loader):
        ttt=self.parameter['train_num']
        test_all=False
        # training by epoch
        list1=[]
        list2=[]
        list3=[]
        list4=[]
        list5=[]
        list6=[]
        list7=[]
        list8=[]
        list9=[]
        list10=[]
        list11=[]
        list12=[]
        pbar = tqdm(range(self.epoch))
        if not os.path.isdir(os.path.join(self.parameter['dataset'], 'train_'+str(ttt))):
            os.makedirs(os.path.join(self.parameter['dataset'], 'train_'+str(ttt)))
        for e in pbar:
            
            t1 = time.time()
            train_task, curr_rel = pretrain2_data_loader.next_batch()
            query_ans_list, negative_ans_list, query_sim_train, negative_sim_train = self.model2_ex(train_task, iseval=False, is_eval_loss = True , curr_rel=curr_rel)
            
            self.optimizer_model2.zero_grad()

            loss_model2 = torch.nn.MarginRankingLoss(self.margin)(query_sim_train, negative_sim_train, torch.Tensor([1]).squeeze(-1).to(self.device))
            loss =  loss_model2
            loss.backward()
            self.optimizer_model2.step()
            
            self.scheduler_model2.step()
            
            
            a=query_sim_train.cpu().detach().numpy()
            b=negative_sim_train.cpu().detach().numpy()
            
            list1.append(a)
            list2.append(b)
            
            if e==0:
                list3.append(a)
                list4.append(b)
            else:
                list3.append(list3[-1]+0.1*(list1[-1]-list3[-1]))
                list4.append(list4[-1]+0.1*(list2[-1]-list4[-1]))
            
            best=0
            if (e+1) % (10*self.checkpoint_epoch) == 0 and (e+1) != 0:
                
                if test_all==True:
                    torch.save(self.model2_ex.state_dict(), os.path.join(self.parameter['dataset'], 'train_'+str(ttt), 'checkpoint' + '.ckpt'))
                    #mrr=self.model2(test_data_loader_ranktail4, (e+1), istest=True)
                    #if mrr>=best:
                    #    best=mrr
                    #    torch.save(self.model2_ex.state_dict(), os.path.join(self.parameter['dataset'], 'train_'+str(ttt), 'best' + '.ckpt'))
                else:
                    torch.save(self.model2_ex.state_dict(), os.path.join(self.parameter['dataset'], 'train_'+str(ttt), 'checkpoint' + '.ckpt'))
         
            
            if (e+1) % self.checkpoint_epoch == 0 and (e+1) != 0:#for figure
                plt.figure()
                l1,=plt.plot(range(1,len(list1)+1),list1,alpha=0.3,color='blue',label='up')
                l2,=plt.plot(range(1,len(list2)+1),list2,alpha=0.3,color='red',label='down')
                l3,=plt.plot(range(1,len(list3)+1),list3,alpha=1,color='blue',label='up')
                l4,=plt.plot(range(1,len(list4)+1),list4,alpha=1,color='red',label='down')
                plt.legend(handles=[l1,l2,],labels=['pos_model2','neg_model2'],loc='best') 
                plt.xlabel("epoch")
                plt.ylabel("score")
                plt.title("model2")
                plt.savefig(os.path.join(self.parameter['dataset'], 'train_'+str(ttt)+'_model2_score.png'))
                plt.close()
                
        pass
    