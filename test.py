## Init
from txgnn import TxData, TxGNN, TxEval

# Download/load knowledge graph dataset
TxData = TxData(data_folder_path = './data')
TxData.prepare_split(split = 'complex_disease', seed = 42)

TxGNN = TxGNN(data = TxData, 
              weight_bias_track = False,
              proj_name = 'TxGNN', # wandb project name
              exp_name = 'TxGNN', # wandb experiment name
              device = 'cuda:0' # define your cuda device
              )

# Initialize a new model
TxGNN.model_initialize(n_hid = 100, # number of hidden dimensions
                      n_inp = 100, # number of input dimensions
                      n_out = 100, # number of output dimensions
                      proto = True, # whether to use metric learning module
                      proto_num = 3, # number of similar diseases to retrieve for augmentation
                      attention = False, # use attention layer (if use graph XAI, we turn this to false)
                      sim_measure = 'all_nodes_profile', # disease signature, choose from ['all_nodes_profile', 'protein_profile', 'protein_random_walk']
                      agg_measure = 'rarity', # how to aggregate sim disease emb with target disease emb, choose from ['rarity', 'avg']
                      num_walks = 200, # for protein_random_walk sim_measure, define number of sampled walks
                      path_length = 2, # for protein_random_walk sim_measure, define path length
                      llm = False)

TxGNN.pretrain(n_epoch = 2, 
              learning_rate = 1e-3,
              batch_size = 1024, 
              train_print_per_n = 20)

# TxGNN.save_model('./pretrain_2')

TxGNN.finetune(n_epoch = 500, 
               learning_rate = 5e-4,
               train_print_per_n = 5,
               valid_per_n = 20)

TxGNN.save_model('./vanilla_pretrain_500')

TxEval = TxEval(model = TxGNN)

result = TxEval.eval_disease_centric(disease_idxs = 'test_set', 
                                     show_plot = False, 
                                     verbose = True, 
                                     save_result = True,
                                     return_raw = False,
                                     save_name = './vanilla_pretrain_500_eval.pkl')

# result = TxEval.eval_disease_centric(disease_idxs = [9907.0, 12787.0], 
#                                      relation = 'indication', 
#                                      save_result = True,
#                                      return_raw = True,
#                                      save_name = './test_eval_result.txt')
