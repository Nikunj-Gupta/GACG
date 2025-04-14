import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATv2Conv 
from components.tgat_module import TGANMARL 
from components.tgat_graph import NeighborFinder 
import random 
import math


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.gat = GATv2Conv(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim, heads=1, concat=False) 
        
        ngh_finder = NeighborFinder(adj_list=[[] for _ in range(self.n_agents + 1)]) 
        self.tgan = TGANMARL(ngh_finder, self.args.rnn_hidden_dim) 

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        
        combined_dim = self.state_dim + self.n_agents * self.args.rnn_hidden_dim

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(combined_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))
        
    def get_edge_index(self, N, type="full"): # need an initial graph construction 
        if type == "line": 
            edges = [[i, i + 1] for i in range(N - 1)]  # # arrange agents in a line 
        elif type == "full": 
            edges = [[(j, i + j + 1) for i in range(N - j - 1)] for j in range(N - 1)]
            edges = [e for l in edges for e in l] 
        elif type == 'cycle':    # arrange agents in a circle
            edges = [(i, i + 1) for i in range(N - 1)] + [(N - 1, 0)] 
        elif type == 'star':     # arrange all agents in a star around agent 0
            edges = [(0, i + 1) for i in range(N - 1)] 
        edge_index = th.tensor(edges).T # # arrange agents in a line   
        # return edge_index.to(self.device) 
        return edge_index

    def generate_edges_with_reset_timesteps_no_interlinks(self, N, g, k, t, neighbor_table):
        """
        Generate edges for fully connected subgraphs with timesteps resetting after every t subgraphs. 
        Removes inter-subgraph edges when the timestep resets. 

        :param N: Total number of nodes
        :param g: Number of agents/nodes per timestep 
        :param k: Number of past iterations considered
        :param t: Reset interval for timesteps
        :return: Tuple (sorted_edges, timestep_values)
        """
        edges = set()  # To store unique edges
        timesteps = {}  # Dictionary to store edge timesteps
        
        k_past_self = int(10/10 * t)
        
        k_past_neighbors = int(t/ (math.log(t) ** 2))    
        
        # # graphv2 
        for batch in range(int((N/g)/t)): 
            start_node = batch*(g*t)
            for reverse_timestep in range(t):
                timestep = t -reverse_timestep -1
                for node in range(start_node, start_node  + timestep*g  ):
                    for j in range(max(0,timestep-k_past_neighbors), timestep+1):
                        for neighbor in neighbor_table[node]: 
                            past_neighbor = neighbor - (timestep- j) * g
                            if (past_neighbor>g-1 and neighbor> g-1 and neighbor-g>= start_node ):
                                edge = (node, past_neighbor) 
                                edges.add(edge)
                                timesteps[edge] = int(node/g)
                                
        # # graphv3  
        # for batch in range(int((N/g)/t)): 
        #     start_node = batch*(g*t)
        #     for reverse_timestep in range(t):
        #         timestep = t -reverse_timestep -1
        #         for node in range(start_node, start_node  + timestep*g  ):
        #             for j in range(max(0,timestep-k_past_neighbors), timestep+1):
        #                 past_node = node - (timestep- j) * g
        #                 if (past_node > g-1):
        #                     for neighbor in neighbor_table[past_node]:
        #                         if (past_node>g-1 and neighbor> g-1 and neighbor-g>= start_node ):
        #                             edge = (node, neighbor) 
        #                             edges.add(edge)
        #                             timesteps[edge] = int(node/g)

        for batch in range(int((N/g)/t)): 
            start_node = batch*(g*t)
            for reverse_timestep in range(t):
                timestep = t -reverse_timestep -1
                for i in range(g): 
                    current_node = start_node + timestep * g + i
                    for j in range(max(0,timestep- k_past_self), timestep+1):
                        past_node = current_node - (timestep- j) * g 
                        if past_node >= start_node: 
                            edge = (past_node, current_node)
                            edges.add(edge)
                            timesteps[edge] = timestep
                            
        sorted_edges = sorted(edges)  # Sort edges for consistency
        timestep_values = [timesteps[edge] for edge in sorted_edges]  # Extract timesteps in sorted order
        return sorted_edges, timestep_values

    def sample_edges(self, edges, timesteps, s):
        """
        Randomly samples s edges and their corresponding timesteps.

        :param edges: List of edges (tuples)
        :param timesteps: List of timesteps corresponding to the edges
        :param s: Number of samples to draw
        :return: List of sampled edges and their corresponding timesteps
        """
        sampled_indices = random.sample(range(len(edges)), min(s, len(edges)))  # Ensure we don't sample more than available
        sampled_edges = [edges[i] for i in sampled_indices]
        sampled_timesteps = [timesteps[i] for i in sampled_indices]
        
        return sampled_edges, sampled_timesteps

    def forward(self, agent_qs, states, hidden_states=None):
        bs = agent_qs.size(0)
        ts =  hidden_states.size(1)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        
        hidden_states = hidden_states.reshape(-1, self.args.rnn_hidden_dim) 
            
        neighbor_table = {i: [] for i in range(bs * ts * self.n_agents)}
        
        static_edges = set()
        
        max_node = bs * ts * self.n_agents  # hidden_states.shape[0] 
        for timestep in range(bs*ts):
            for i in range(self.n_agents):
                node = i + timestep * self.n_agents
                for neighbor in range(node, node + self.n_agents):
                    if neighbor < max_node:  # fix here
                        edge = (node, neighbor)
                        static_edges.add(edge)
        
        sorted_static_edges = sorted(static_edges)    
        sorted_static_edges = th.tensor(sorted_static_edges).T.to(hidden_states.device)
        hidden_states, (edge_index, attention_weights) = self.gat(hidden_states, edge_index=sorted_static_edges, return_attention_weights=True)

        timestep_per_edge = edge_index[0] // self.n_agents 
        
        filtered_edges = []
        
        for timestep in range(bs * ts):
        
            t_mask = (timestep_per_edge == timestep)
            
            if t_mask.sum() == 0:
                continue  # Skip if there are no edges for this timestep
            
            # Get the indices for these edges
            t_attention = attention_weights[t_mask]
                
            median_val = th.quantile(t_attention, 0.5)
            
            keep_mask = (t_attention >= median_val)
            
            t_indices = keep_mask.nonzero(as_tuple=True)[0]
            
            for i in t_indices:
                filtered_edges.append(edge_index[:, i])
                    
        filtered_edge_index = th.stack(filtered_edges, dim=1)
        
        for src, dst in filtered_edge_index.t().tolist():
            if (src != dst):
                neighbor_table[src].append(dst)
                neighbor_table[dst].append(src)
                
                #comment

        edges, timesteps = self.generate_edges_with_reset_timesteps_no_interlinks(bs * ts * self.n_agents, self.n_agents, 3, ts, neighbor_table) # N, g, k, t 
        tgat_batch = 4 
        for _ in range(tgat_batch):  
            sampled_edges, sampled_timesteps = self.sample_edges(edges, timesteps, bs * ts * self.n_agents) 
            sampled_edges = th.tensor(sampled_edges).T 
            # hidden_states, (edges, weights) = self.gat(hidden_states, edge_index=edges, return_attention_weights=True)
            train_src_l = sampled_edges[0].tolist() 
            train_dst_l = sampled_edges[1].tolist() 
            # train_e_idx_l = list(range(1, bs * ts * self.n_agents + 1)) 
            train_e_idx_l = list(range(1, sampled_edges.shape[1] + 1)) 
            train_ts_l = sampled_timesteps 

            adj_list = [[] for _ in range(bs * ts * self.n_agents + 1)] 
            for src, dst, eidx, tss in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l): 
                adj_list[src].append((dst, eidx, tss))
                adj_list[dst].append((src, eidx, tss)) 
            ngh_finder = NeighborFinder(adj_list) 
            self.tgan.ngh_finder = ngh_finder 
            hidden_states = self.tgan(n_feat_th=hidden_states, src_idx_l=np.array(train_src_l), cut_time_l=np.array(train_ts_l)) 

        hidden_states = hidden_states.reshape(-1, self.n_agents, self.args.rnn_hidden_dim)
        
        hidden_flat = hidden_states.view(bs, ts, -1)  # shape: (bs, ts, n * hidden_state_dim)
        
        combined_states = torch.cat([states, hidden_flat], dim=-1)  # shape: (bs, ts, state_dim + n*hidden_state_dim)
        
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(combined_states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
