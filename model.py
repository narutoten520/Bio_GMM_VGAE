import torch
import numpy as np
import torch.nn as nn
from torch.nn import Linear
import scipy.sparse as sp
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import StepLR
from preprocess import sparse_to_tuple,random_uniform_init,q_mat,generate_unconflicted_data_index,mclust_R,refine_label,\
    cal_metrics,fix_seed
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.decomposition import PCA

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = random_uniform_init(input_dim, output_dim) 
        self.activation = activation
        
    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs
    
class Encoders(nn.Module):
    def __init__(self, **kwargs):
        super(Encoders, self).__init__()
        self.num_neurons = kwargs['hidden_dim']
        self.num_features = kwargs['num_features']
        self.embedding_size = kwargs['embed_dim']
        self.nClusters = kwargs['nClusters']
        self.position = kwargs['position']
        self.random_seed = kwargs['random_seed']
        fix_seed(self.random_seed)
        
        # VGAE training parameters
        self.base_gcn = GraphConvSparse( self.num_features, self.num_neurons)
        self.gcn_mean = GraphConvSparse( self.num_neurons, self.embedding_size, activation = lambda x:x)
        self.gcn_logstddev = GraphConvSparse( self.num_neurons, self.embedding_size, activation = lambda x:x)
        
        # GMM training parameters    
        self.pi = nn.Parameter(torch.ones(self.nClusters)/self.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size),requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size),requires_grad=True)
        
    def encode(self, x_features, adj):
        hidden = self.base_gcn(x_features, adj)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)
        gaussian_noise = torch.randn(x_features.size(0), self.embedding_size)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return self.mean, self.logstd ,sampled_z
            
    @staticmethod
    def decode(z):
        A_pred = torch.sigmoid(torch.matmul(z,z.t()))
        return A_pred
                                  
    def pretrain(self, adj, features, adj_label, weight_tensor, norm, epochs, lr, save_path, section_id):
        opti = Adam(self.parameters(), lr=lr)
        epoch_bar = tqdm(range(epochs))
        gmm = GaussianMixture(n_components = self.nClusters , covariance_type = 'diag')
        for _ in epoch_bar:
            opti.zero_grad()
            _,_, z = self.encode(features, adj)
            x_ = self.decode(z)          
            loss = norm * F.binary_cross_entropy(x_.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
            loss.backward()
            opti.step()
        gmm.fit(z.detach().numpy())
        self.pi.data = torch.from_numpy(gmm.weights_)
        self.mu_c.data = torch.from_numpy(gmm.means_)
        self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_))
        self.logstd = self.mean 
        torch.save(self.state_dict(), save_path + section_id + '/model.pk')

                              
    def train(self, adj_norm, adj, features, y, norm, epochs, lr, beta1, beta2 , save_path, section_id):
        self.load_state_dict(torch.load(save_path + section_id + '/model.pk'))
        opti = Adam(self.parameters(), lr=lr , weight_decay = 1e-4)
        lr_s = StepLR(opti, step_size=10, gamma=0.9) 
        epoch_bar = tqdm(range(epochs))
        epoch_stable = 0
        previous_unconflicted = []
        previous_conflicted = []
        ari_list=[]
        ari_best = 0 
        loss_list = []
        for epoch in epoch_bar:
            opti.zero_grad()
            z_mu, z_sigma2_log, emb = self.encode(features, adj_norm) 
            x_ = self.decode(emb)          
            if epoch % 5 == 0 :
                unconflicted_ind, conflicted_ind = generate_unconflicted_data_index(emb, self.mu_c, beta1, beta2)
            if epoch == 0:
                adj, adj_label, weight_tensor =  self.update_graph(adj, y, emb, unconflicted_ind, conflicted_ind)
            if len(previous_unconflicted) < len(unconflicted_ind) :
                z_mu = z_mu[unconflicted_ind]
                z_sigma2_log = z_sigma2_log[unconflicted_ind]
                emb_unconf = emb[unconflicted_ind]
                emb_conf = emb[conflicted_ind]
                previous_conflicted = conflicted_ind
                previous_unconflicted = unconflicted_ind     
            else :
                epoch_stable += 1
                z_mu = z_mu[previous_unconflicted]
                z_sigma2_log = z_sigma2_log[previous_unconflicted]
                emb_unconf = emb[previous_unconflicted]
                emb_conf = emb[previous_conflicted]
            if epoch_stable >= 5:
                epoch_stable = 0
                beta1 = beta1 * 0.95
                beta2 = beta2 * 0.85
            if epoch % 2 == 0 and epoch <= 100:
                adj, adj_label, weight_tensor =  self.update_graph(adj, y, emb, unconflicted_ind, conflicted_ind)
            loss, loss1, elbo_loss = self.ELBO_Loss(features, adj_norm, x_, adj_label.to_dense().view(-1), weight_tensor, norm, z_mu , z_sigma2_log, emb_unconf)            
            
            epoch_bar.write('Loss={:.4f},  ELBO Loss={:.4f}'.format(loss.detach().numpy(), elbo_loss.detach().numpy()))
            
            y_prediction = self.prediction(emb.detach().numpy())
            metrics = cal_metrics(y, y_prediction) 
            print('NMI=%f, ARI=%f' % (metrics["nmi"], metrics["ari"]))
            ari_list.append(metrics["ari"])
            loss_list.append(elbo_loss.detach().numpy())
            
            # save the best prediction labels
            if metrics["ari"]>ari_best:
                ari_best = metrics["ari"]
                pred_labels = y_prediction
                Rex = x_.detach().numpy()
                Rex[Rex<0] = 0
                embedding = emb.detach().numpy()
            elbo_loss.backward()
            opti.step()
            lr_s.step()
        # best_ari = np.max(ari_list)
        # print(f"The best ARI is {best_ari}")
        return pred_labels,ari_list,Rex,embedding,loss_list
    
    def update_graph(self, adj, labels, emb, unconf_indices, conf_indices):
        k = 0
        y_pred = self.predict(emb)
        # y_pred = self.prediction(emb.detach().numpy())
        emb_unconf = emb[unconf_indices]
        adj = adj.tolil()        
        idx = unconf_indices[self.generate_centers(emb_unconf)]    
        for i, k in enumerate(unconf_indices):
            # adj_k = adj[k].tocsr().indices
            adj_k = adj[[k],:].tocsr().indices
            if not(np.isin(idx[i], adj_k)) and (y_pred[k] == y_pred[idx[i]]) :
                adj[k, idx[i]] = 1
            for j in adj_k:
                if np.isin(j, unconf_indices) and (np.isin(idx[i], adj_k)) and (y_pred[k] != y_pred[j]):
                    adj[k, j] = 0
        adj = adj.tocsr()
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                    torch.FloatTensor(adj_label[1]),
                                    torch.Size(adj_label[2]))
        weight_mask = adj_label.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum() 
        weight_tensor[weight_mask] = pos_weight_orig
        return adj, adj_label, weight_tensor
    
    def generate_centers(self, emb_unconf):
        y_pred = self.predict(emb_unconf)
        
        nn = NearestNeighbors(n_neighbors= 1, algorithm='kd_tree').fit(emb_unconf.detach().numpy())
        _, indices = nn.kneighbors(self.mu_c.detach().numpy())
        return indices[y_pred]
    
    def ELBO_Loss(self, features, adj, x_, adj_label, weight_tensor, norm, z_mu, z_sigma2_log, emb, L=1):
        pi = self.pi
        mu_c = self.mu_c
        log_sigma2_c = self.log_sigma2_c
        det = 1e-2
        Loss = 1e-2 * norm * F.binary_cross_entropy(x_.view(-1), adj_label, weight = weight_tensor)
        Loss = Loss * features.size(0)
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(emb,mu_c,log_sigma2_c))+det
        yita_c = yita_c / (yita_c.sum(1).view(-1,1))
        KL1 = 0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))
        Loss1 = KL1 
        KL2 = torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))
        Loss1 -= KL2
        return Loss, Loss1, Loss+Loss1
    
    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.nClusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    def gaussian_pdf_log(self,x,mu,log_sigma2):
        c = -0.5 * torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1)
        return c

    def predict(self, z):
        pi = self.pi
        log_sigma2_c = self.log_sigma2_c  
        mu_c = self.mu_c
        det = 1e-2
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det
        yita = yita_c.detach().numpy()
        return np.argmax(yita, axis=1)
    
    def prediction(self,z):
        pca20 = PCA(n_components=27)
        pca_z = pca20.fit_transform(z)
        pred_mclust = mclust_R(embedding=pca_z,num_cluster=self.nClusters)
        pred_mclust = refine_label(pred_mclust, self.position, radius=50)
        return pred_mclust

