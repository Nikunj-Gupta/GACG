import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# from torch_geometric.nn import GATConv 
from torch_geometric.nn import GATv2Conv 
from components.tgat_module import TGANMARL 
from components.tgat_graph import NeighborFinder 
import random 

class QTranBase(nn.Module):
    def __init__(self, args):
        super(QTranBase, self).__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.arch = self.args.qtran_arch # QTran architecture

        self.embed_dim = args.mixing_embed_dim
        self.gat = GATv2Conv(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim, heads=1, concat=False) 

        ngh_finder = NeighborFinder(adj_list=[[] for _ in range(self.n_agents + 1)]) 
        self.tgan = TGANMARL(ngh_finder, self.args.rnn_hidden_dim) 

        # Q(s,u)
        if self.arch == "coma_critic":
            # Q takes [state, u] as input
            q_input_size = self.state_dim + (self.n_agents * self.n_actions)
        elif self.arch == "qtran_paper":
            # Q takes [state, agent_action_observation_encodings]
            q_input_size = self.state_dim + self.args.rnn_hidden_dim + self.n_actions
        else:
            raise Exception("{} is not a valid QTran architecture".format(self.arch))

        if self.args.network_size == "small":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.embed_dim, 1))

            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.embed_dim, 1))
            ae_input = self.args.rnn_hidden_dim + self.n_actions
            self.action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(ae_input, ae_input))
        elif self.args.network_size == "big":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.embed_dim, 1))
            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.embed_dim, 1))
            ae_input = self.args.rnn_hidden_dim + self.n_actions
            self.action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(ae_input, ae_input))
        else:
            assert False
    
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
        
        k_past_neighbors = int(t/ math.log(t) ** 3)
        
        for batch in range(int((N/g)/t)): 
            start_node = batch*(g*t)
            for node in range(start_node, start_node  + t*g  ):
                    for neighbor in neighbor_table[node]: 
                        if (node>g-1 and neighbor> g-1 and neighbor-g>= start_node ):
                            edge = (node, neighbor-g) 
                            edges.add(edge)
                            timesteps[edge] = int(node/g)

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

    def forward(self, batch, hidden_states, actions=None):
        bs = batch.batch_size
        ts = batch.max_seq_length

        states = batch["state"].reshape(bs * ts, self.state_dim)

        if self.arch == "coma_critic":
            if actions is None:
                # Use the actions taken by the agents
                actions = batch["actions_onehot"].reshape(bs * ts, self.n_agents * self.n_actions)
            else:
                # It will arrive as (bs, ts, agents, actions), we need to reshape it
                actions = actions.reshape(bs * ts, self.n_agents * self.n_actions)
            inputs = th.cat([states, actions], dim=1)
        elif self.arch == "qtran_paper":
            if actions is None:
                # Use the actions taken by the agents
                actions = batch["actions_onehot"].reshape(bs * ts, self.n_agents, self.n_actions)
            else:
                # It will arrive as (bs, ts, agents, actions), we need to reshape it
                actions = actions.reshape(bs * ts, self.n_agents, self.n_actions)
            
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
            sorted_static_edges = th.tensor(sorted_static_edges).T 
            hidden_states, (edge_index, attention_weights) = self.gat(hidden_states, edge_index=sorted_static_edges, return_attention_weights=True)
            
            min_val = attention_weights.min().item()
            max_val = attention_weights.max().item()
            
            threshold = (min_val + max_val)/2

            timestep_per_edge = edge_index[0] // self.n_agents 
            
            filtered_edges = []
            
            for timestep in range(bs * ts):
            
                t_mask = (timestep_per_edge == timestep)
                
                if t_mask.sum() == 0:
                    continue  # Skip if there are no edges for this timestep
                
                # Compute the mean attention weight A_t for the current timestep
                M = attention_weights[t_mask].mean()
                
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
            agent_state_action_input = th.cat([hidden_states, actions], dim=2)
            agent_state_action_encoding = self.action_encoding(agent_state_action_input.reshape(-1, self.args.rnn_hidden_dim + self.n_actions)).reshape(-1, self.n_agents, self.args.rnn_hidden_dim + self.n_actions)
            agent_state_action_encoding = agent_state_action_encoding.sum(dim=1) # Sum across agents

            inputs = th.cat([states, agent_state_action_encoding], dim=1)

        q_outputs = self.Q(inputs)

        states = batch["state"].reshape(-1, self.state_dim)
        v_outputs = self.V(states)

        return q_outputs, v_outputs