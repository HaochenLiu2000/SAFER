Our code is based on the code of CSR (https://github.com/snap-stanford/csr) and we use the same dataset as CSR. The dataset, requirements, and data preparation follow the setting of CSR.

To train our SAFER model on dataset(NELL, FB15K-237, ConceptNet):

python main.py --use_atten True --use_pretrain_node_emb True --dataset <dataset> --device 0 --step pretrain2 -prev_state_dir_model2 <dataset>/train_1/checkpoint.ckpt --train_num 1 -epo 20000 

To test the trained SAFER model:

python main.py --use_atten True --use_pretrain_node_emb True --dataset <dataset> --device 0 --step model2 -prev_state_dir_model2 <dataset>/train_1/checkpoint.ckpt
