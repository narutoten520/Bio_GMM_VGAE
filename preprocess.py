import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import ot
import os
import random
from torch.backends import cudnn
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import torch
import scanpy as sc
from sklearn.decomposition import PCA
import torch.nn as nn

def read_adata(section):
    section_id = section
    file_fold = '/home/tengliu/Paper6-NC/stAA-main/stAA-model-ARGVA-12samples/DLPFC_data/'+section_id #please replace 'file_fold' with the download path
    tissue_positions_list = pd.read_csv(file_fold+"/spatial/tissue_positions_list.txt")
    tissue_positions_list.to_csv(file_fold+"/spatial/tissue_positions_list.csv",index=None)
    adata = sc.read_visium(file_fold, count_file=section_id+'_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()

    # add ground_truth
    df_meta = pd.read_csv(file_fold + '/cluster_labels_'+section_id+'.csv')
    df_meta_layer = df_meta["ground_truth"]
    adata.obs['ground_truth'] = df_meta_layer.values
    return adata

def random_uniform_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

def q_mat(X, centers, alpha=1.0):
    X = X.detach().numpy()
    centers = centers.detach().numpy()
    if X.size == 0:
        q = np.array([])
    else:
        q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(X, 1) - centers), axis=2) / alpha))
        q = q**((alpha+1.0)/2.0)
        q = np.transpose(np.transpose(q)/np.sum(q, axis=1))
    return q

def generate_unconflicted_data_index(emb, centers_emb, beta1, beta2):
    unconf_indices = []
    conf_indices = []
    q = q_mat(emb, centers_emb, alpha=1.0) # 相似度矩阵 q
    confidence1 = np.zeros((q.shape[0],))
    confidence2 = np.zeros((q.shape[0],))
    a = np.argsort(q, axis=1)
    for i in range(q.shape[0]):
        confidence1[i] = q[i,a[i,-1]]
        confidence2[i] = q[i,a[i,-2]]
        if (confidence1[i]) > beta1 and (confidence1[i] - confidence2[i]) > beta2:
            unconf_indices.append(i)
        else:
            conf_indices.append(i)
    unconf_indices = np.asarray(unconf_indices, dtype=int)
    conf_indices = np.asarray(conf_indices, dtype=int)
    return unconf_indices, conf_indices

def adata_hvg(adata,gene_num=3000):
    adata.var_names_make_unique()
    #Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=gene_num) 
    sc.pp.normalize_total(adata, target_sum=1e4) ##normalized data
    sc.pp.log1p(adata)  #log-transformed data
    adata = adata[:, adata.var['highly_variable']]
    return adata

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def adj_matrix(adata, n_neighbors=5):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    adata.obsm['distance_matrix'] = distance_matrix
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj
    print('Graph constructed!')
    
def adj_matrix_KNN(adata, n_neighbors=5):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)  
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj
    print('Graph constructed!')

def feature_matrix(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat_mat = adata_Vars.X.toarray()[:, ]
    else:
       feat_mat = adata_Vars.X[:, ]
       
    adata.obsm['feat_mat'] = feat_mat
   
def dlpfc_gt_labels(adata):
    if 'ground_truth' not in adata.obs.keys():
        raise ValueError(
            "Please load the ground truth of ST data first!")
    adata.obs["ss_label"] = adata.obs['ground_truth']
    ss_label = adata.obs["ss_label"].to_list()
    label_dict = {'Layer_1': 0, 'Layer_2': 1, 'Layer_3': 2, 'Layer_4': 3, 'Layer_5': 4, 'Layer_6': 5, 'WM': 6}
    ss_label = [label_dict.get(label, label) for label in ss_label]
    ss_label = np.array(ss_label, dtype=float)
    ss_label = np.nan_to_num(ss_label)
    adata.uns["dlpfc_gt_labels"] = ss_label
    
def cal_metrics(true, pred):
    from sklearn.metrics import adjusted_rand_score, completeness_score, homogeneity_score, v_measure_score
    label_df = pd.DataFrame({"True": true,"Pred": pred}).dropna()
    completeness = completeness_score(label_df["True"], label_df["Pred"])
    hm = homogeneity_score(label_df["True"], label_df["Pred"])
    nmi = v_measure_score(label_df["True"], label_df["Pred"])
    ari = adjusted_rand_score(label_df["True"], label_df["Pred"])
    metrics = {}
    metrics["ari"] = ari
    metrics["completeness"] = completeness
    metrics["homogeneity"] = hm
    metrics["nmi"] = nmi
    return metrics

def dopca(X, dim=10):
    pcaten = PCA(n_components=dim, random_state=42)
    X_10 = pcaten.fit_transform(X)
    return X_10

def mclust_R(embedding, num_cluster, modelNames='EEE', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(
        embedding), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    mclust_res = mclust_res.astype('int')
    # mclust_res = mclust_res.astype('category')
    return mclust_res

def refine_label(label, position, radius=50):
    new_type = []
    distance = ot.dist(position, position, metric='euclidean')
    n_cell = distance.shape[0]
    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, radius+1):
            neigh_type.append(label[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
    new_type = [str(i) for i in list(new_type)]
    return new_type

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
def mclust_initial_label(adata, n_clusters, refine=True, method="mclust"):
    if 'highly_variable' not in adata.var.keys():
        adata_hvg(adata) 
    feature_matrix(adata)
    features = np.asarray(adata.obsm['feat_mat'])
    pca_input = dopca(features, dim = 20)
    if method == "mclust":
        pred = mclust_R(embedding=pca_input, num_cluster=n_clusters)
    if method == "louvain":
        adata.obsm["pca"] = pca_input
        sc.pp.neighbors(adata, n_neighbors=50, use_rep="pca")
        sc.tl.louvain(adata, resolution=n_clusters, random_state=0)
        pred=adata.obs['louvain'].astype(int).to_numpy()
    if refine:
        pred = refine_label(pred, adata.obsm["spatial"], radius=60)
    pred = list(map(int, pred))
    adata.uns["mclust_labels"] = np.array(pred)