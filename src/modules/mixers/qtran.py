import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv 
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
        self.gat = GATConv(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim, heads=1, concat=False) 

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

    def generate_edges_with_reset_timesteps_no_interlinks(self, N, k, t):
        """
        Generate edges for fully connected subgraphs with timesteps resetting after every t subgraphs.
        Removes inter-subgraph edges when the timestep resets.

        :param N: Total number of nodes
        :param k: Number of nodes per subgraph
        :param t: Reset interval for timesteps
        :return: Tuple (sorted_edges, timestep_values)
        """
        edges = set()  # To store unique edges
        timesteps = {}  # Dictionary to store edge timesteps

        num_subgraphs = N // k  # Number of subgraphs

        for s in range(num_subgraphs):
            start = s * k  # Start index for the subgraph
            current_timestep = (s % t) + 1  # Reset after every t subgraphs

            # Generate fully connected subgraph edges
            subgraph = [(start + j, start + i + j + 1) for j in range(k - 1) for i in range(k - j - 1)]
            
            # Assign timestep for the current subgraph edges
            for edge in subgraph:
                edges.add(edge)
                timesteps[edge] = current_timestep

            # Add inter-subgraph connections (previous subgraph to current) only if no reset occurs
            if s > 0 and s % t != 0:  # Avoid adding interlinks when timestep resets
                for i in range(k):  # Connect each node to its corresponding node in the previous subgraph
                    prev_node = (s - 1) * k + i
                    curr_node = start + i
                    if prev_node < curr_node:  # Avoid duplicate edges
                        edges.add((prev_node, curr_node))
                        timesteps[(prev_node, curr_node)] = current_timestep

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

            edges, timesteps = self.generate_edges_with_reset_timesteps_no_interlinks(bs * ts * self.n_agents, self.n_agents, ts) 
            sampled_edges, sampled_timesteps = self.sample_edges(edges, timesteps, bs * ts * self.n_agents)
            edges = th.tensor(edges).T 
            sampled_edges = th.tensor(sampled_edges).T 

            hidden_states, (edges, weights) = self.gat(hidden_states, edge_index=edges, return_attention_weights=True) 

            train_src_l = sampled_edges[0].tolist() 
            train_dst_l = sampled_edges[1].tolist() 
            train_e_idx_l = list(range(1, bs * ts * self.n_agents + 1)) 
            train_ts_l = sampled_timesteps 

            adj_list = [[] for _ in range(bs * ts * self.n_agents + 1)] 
            for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l): 
                adj_list[src].append((dst, eidx, ts))
                adj_list[dst].append((src, eidx, ts)) 
            ngh_finder = NeighborFinder(adj_list) 
            self.tgan.ngh_finder = ngh_finder 
            hidden_states = self.tgan.forward(n_feat_th=hidden_states, src_idx_l=np.array(train_src_l), cut_time_l=np.array(train_ts_l)) 

            hidden_states = hidden_states.reshape(-1, self.n_agents, self.args.rnn_hidden_dim)
            agent_state_action_input = th.cat([hidden_states, actions], dim=2)
            agent_state_action_encoding = self.action_encoding(agent_state_action_input.reshape(-1, self.args.rnn_hidden_dim + self.n_actions)).reshape(-1, self.n_agents, self.args.rnn_hidden_dim + self.n_actions)
            agent_state_action_encoding = agent_state_action_encoding.sum(dim=1) # Sum across agents

            inputs = th.cat([states, agent_state_action_encoding], dim=1)

        q_outputs = self.Q(inputs)

        states = batch["state"].reshape(-1, self.state_dim)
        v_outputs = self.V(states)

        return q_outputs, v_outputs

