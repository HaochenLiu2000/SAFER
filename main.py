from trainer import *
from params import *
import json
import models
from load_kg_dataset import *
import pdb


if __name__ == '__main__':
    params = get_params()

    print("---------Parameters---------")
    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")
    if params['real']:
        n_query = 10
        kind = "union_prune_plus"
        if params['dataset'] in ['NELL']: 
            hop = 2
        elif params['dataset'] in ['FB15K-237', 'ConceptNet']:
            hop = 1
        else:
            assert False
    #prepare datasets
    test_data_loader_ranktail4=PairSubgraphsFewShotDataLoader_model2(SubgraphFewshotDatasetRankTail4(params["data_path"], hop = hop, shot = params['few'], n_query = n_query, dataset=params['dataset'], mode="test", rev = params['rev'], kind=kind, use_fix2 = params['fix2'], num_rank_negs = params['num_rank_negs'], inductive = params['inductive'], orig_test= params['orig_test']), batch_size=  params["rank_tail_batch_size"])
    pretrain2_data_loader = PairSubgraphsFewShotDataLoader_model2(SubgraphFewshotDataset(params["data_path"], shot = params['few'], dataset=params['dataset'], mode="pretrain", rev = params['rev'], kind=kind, hop=hop, use_fix2 = params['fix2'], num_rank_negs = params['num_rank_negs'], orig_test= params['orig_test']), batch_size= 1, shuffle = True)
    train_data_loader = PairSubgraphsFewShotDataLoader(SubgraphFewshotDataset(params["data_path"], shot = params['few'], dataset=params['dataset'], mode="train", rev = params['rev'], kind=kind, hop=hop, use_fix2 = params['fix2'], num_rank_negs = params['num_rank_negs'], inductive = params['inductive'], orig_test= params['orig_test'] ), batch_size= params["batch_size"], shuffle = True)
    #data_loaders = [test_data_loader_ranktail4, pretrain2_data_loader]
    trainer = Trainer(train_data_loader,params)

    #test
    if params['step'] == 'model2': 
        data = trainer.model2(test_data_loader_ranktail4, istest=True)
        #train
    elif params['step'] == 'pretrain2':
        trainer.pretrain2(pretrain2_data_loader)