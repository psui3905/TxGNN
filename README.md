# Bridging the Unknown: Can LLMs Revolutionize Drug Repurposing by Enriching the Molecular Landscape of Uncharacterized Diseases?

In this work, we introduces a novel statistical methodology utilizing Large Language Models (LLMs) to enhance Similarity Disease Search based on TxGNN. We show that LLMs can be used to enrich the molecular landscape of uncharacterized diseases, and thus improve the performance of TxGNN. We also show that LLMs can be used to generate novel hypotheses for drug repurposing.

### Original Work of TxGNN [https://www.medrxiv.org/content/10.1101/2023.03.19.23287458](https://www.medrxiv.org/content/10.1101/2023.03.19.23287458)

### Installation 

```bash
conda create --name txgnn_env python=3.8
conda activate txgnn_env
# Install PyTorch via https://pytorch.org/ with your CUDA versions
conda install -c dglteam dgl-cuda{$CUDA_VERSION}==0.5.2 # checkout https://www.dgl.ai/pages/start.html for more info, as long as it is DGL 0.5.2
pip install TxGNN
```

Note that if you want to use disease-area split, you should also install PyG following [this instruction](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) since some legacy data processing code uses PyG utility functions.

### Evaluation
To evaluate the performance of TxGNN with LLM-basd Similarity Disease Search, you can simply run the following code:

```bash 
conda activate txgnn_env
cd ./TxGNN
python3 test.py <llm-api-key>
```

### Core API Interface
Using the API, you can (1) reproduce the results in our paper and (2) train TxGNN on your own drug repurposing dataset using a few lines of code, and also generate graph explanations. 

```python
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
                      llm = True # whether to use LLMs, default GPT-4
                      )

```

To finetuning on drug-disease relation with metric learning, you can type:

```python
TxGNN.finetune(n_epoch = 500, 
               learning_rate = 5e-4,
               train_print_per_n = 5,
               valid_per_n = 20,
               save_name = finetune_result_path)
```

To save the trained model, you can type:

```python
TxGNN.save_model('./model_ckpt')
```

