import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from preprocess import adata_hvg,adj_matrix,adj_matrix_KNN,feature_matrix,preprocess_graph,sparse_to_tuple, dlpfc_gt_labels,mclust_initial_label
from model import Encoders

class STGMMVE(nn.Module):
    def __init__(self, 
        adata, 
        hidden_dim=240,
        embed_dim=40,
        epochs_pretrain=1000,
        lr_pretrain=0.00001,
        epochs_cluster = 50,
        lr_cluster = 0.0001,
        beta1 = 0.002,
        beta2 = 0.001,
        datatype="10X",
        random_seed = 0,
        activation="ReLU",
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        imputation = False,
        save_path = None,
        nCluster = 0,
        section_id = "151510"
        ):
        super(STGMMVE, self).__init__()
        self.adata = adata.copy()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.epochs_pretrain = epochs_pretrain
        self.lr_pretrain = lr_pretrain
        self.epochs_cluster = epochs_cluster
        self.lr_cluster = lr_cluster
        self.beta1 = beta1
        self.beta2 = beta2
        self.datatype = datatype
        self.random_seed = random_seed
        self.activation = activation
        self.device = device
        self.imputation = imputation
        self.save_path = save_path
        self.section_id = section_id
        
        if self.datatype== "10X":
            self.nClusters = 5 if self.section_id in ['151669', '151670', '151671', '151672'] else 7
        else:
            self.nClusters = nCluster
            
        self.position = self.adata.obsm['spatial']
        gene_num = 2000 if self.section_id in ['151507', '151510', '151670', '151672','151674','151675'] else 3000
        if not self.imputation:
            if 'highly_variable' not in self.adata.var.keys():
                adata_hvg(self.adata,gene_num=gene_num)
        else:
            # sc.pp.calculate_qc_metrics(self.adata, inplace=True)
            # self.adata = self.adata[:,self.adata.var['total_counts']>100]
            # #Normalization
            # sc.pp.normalize_total(self.adata, target_sum=1e4)
            # sc.pp.log1p(self.adata)
            self.adata = adata_hvg(self.adata, gene_num=self.adata.n_obs)
            
        if 'adj' not in self.adata.obsm.keys():
            if self.datatype in ['Stereo', 'Slide']:
                adj_matrix_KNN(self.adata)
            else:
                adj_matrix(self.adata)
                
        if 'feat' not in self.adata.obsm.keys():    
            feature_matrix(self.adata)
        
        adj = csr_matrix(self.adata.obsm["adj"])
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        adj_norm = preprocess_graph(adj)
        pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]), torch.Size(adj_norm[2]))
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]), torch.Size(adj_label[2]))
        features = csr_matrix(self.adata.obsm['feat_mat'])
        features = sparse_to_tuple(features.tocoo())
        num_features = features[2][1]
        features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), torch.FloatTensor(features[1]), torch.Size(features[2]))

        weight_mask_orig = adj_label.to_dense().view(-1) == 1
        weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
        weight_tensor_orig[weight_mask_orig] = pos_weight_orig

        if self.datatype == "10X":
            dlpfc_gt_labels(self.adata)
            true_labels = self.adata.uns["dlpfc_gt_labels"]
            self.true_labels = true_labels
        else:
            if self.section_id in ['mouse_hippo', 'mouse_brain']:
                self.true_labels = self.adata.obs["cluster"]
                
            elif self.section_id == 'breast_cancer':
                self.true_labels = self.adata.obs["ground_truth"]
            elif self.section_id == 'mouse_embryos_E9.5':
                self.true_labels = self.adata.obs["annotation"]   
            else:    
                mclust_initial_label(self.adata, self.nClusters)
                self.true_labels = self.adata.uns["mclust_labels"]
        
        self.adj_norm = adj_norm
        self.adj = adj
        self.features = features
        self.adj_label = adj_label
        self.weight_tensor_orig = weight_tensor_orig
        self.norm = norm
        self.num_features = num_features
               
        # Pretrain the network and save the model
        self.network = Encoders(adj = self.adj_norm, hidden_dim=self.hidden_dim, num_features=self.num_features, embed_dim=self.embed_dim, nClusters=self.nClusters, position=self.position, random_seed=self.random_seed, activation=self.activation)
    
    def pretrain(self):
        # self.network = Encoders(adj = self.adj_norm, hidden_dim=self.hidden_dim, num_features=self.num_features, embed_dim=self.embed_dim, nClusters=self.nClusters, position=self.position, random_seed=self.random_seed, activation=self.activation).to(self.device)
        self.network.pretrain(self.adj_norm, self.features, self.adj_label, self.weight_tensor_orig, self.norm , epochs=self.epochs_pretrain, lr=self.lr_pretrain, save_path=self.save_path, section_id=self.section_id)
        
        # Use the pretrained model to cluster the ST data
    def train_cluster(self):
        # self.network = Encoders(adj = self.adj_norm, hidden_dim=self.hidden_dim, num_features=self.num_features, embed_dim=self.embed_dim, nClusters=self.nClusters, position=self.position, random_seed=self.random_seed, activation=self.activation).to(self.device)
        label,ari,Rex,emb,loss_list = self.network.train(self.adj_norm, self.adj,  self.features, self.true_labels , self.norm, epochs=self.epochs_cluster, lr=self.lr_cluster, beta1=self.beta1, beta2=self.beta2, save_path=self.save_path, section_id=self.section_id)
        
        if self.imputation:
            self.adata.layers['Recons_features'] = Rex
            self.adata.obs['pre_label'] = label
            self.adata.uns["ari_list"] = ari
            self.adata.obsm["embedding"] = emb
            self.adata.uns["loss"] = loss_list
            return  self.adata
        else:
            self.adata.obs['pre_label'] = label
            self.adata.uns["ari_list"] = ari
            self.adata.obsm["embedding"] = emb
            self.adata.uns["loss"] = loss_list
            return  self.adata

