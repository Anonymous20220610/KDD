import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ExpertNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_experts)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)  

class DeepSeekMoE(nn.Module):
    def __init__(self, input_dim, num_experts, output_dim, k=3):
        super(DeepSeekMoE, self).__init__()
        self.num_experts = num_experts
        self.k = k  
        self.experts = nn.ModuleList([ExpertNetwork(input_dim, output_dim) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        batch_size, few, input_dim = x.shape

        gating_weights = self.gating_network(x.view(batch_size * few, -1)) 
        gating_weights = gating_weights.view(batch_size, few, self.num_experts)
        expert_outputs = torch.stack([expert(x.view(-1, input_dim)) for expert in self.experts], dim=1)
        expert_outputs = expert_outputs.view(batch_size, few, self.num_experts, -1)  
        top_k_values, top_k_indices = gating_weights.topk(k=self.k, dim=2, largest=True)  
        top_k_expert_outputs = expert_outputs.gather(2, top_k_indices.unsqueeze(3).expand(-1, -1, -1, expert_outputs.size(3)))  
        weighted_expert_outputs = torch.sum(top_k_expert_outputs * top_k_values.unsqueeze(3), dim=2)  
        weighted_expert_outputs = weighted_expert_outputs.mean(dim=1, keepdim=True) 

        return weighted_expert_outputs.unsqueeze(2) 



