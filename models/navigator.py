import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledSigmoid(nn.Module):
    def __init__(self, scale):
        super(ScaledSigmoid, self).__init__()
        self.scale = scale

    def forward(self, x):
        return torch.sigmoid(self.scale * x)


class Navigator(nn.Module):
    def __init__(self, config):
        super(Navigator, self).__init__()

        self.metric_distance_proj = nn.Linear(1, config.hidden_dim)
        self.metric_angle_proj = nn.Linear(1, config.hidden_dim)

        self.q_proj = nn.Linear(config.hidden_dim * 3, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim * 3, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, 1, bias=False)

        self.time_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.time_estimator_linear_1 = nn.Linear(config.hidden_dim * 2, 64)
        self.time_estimator_linear_2 = nn.Linear(config.hidden_dim, 64)
        self.time_estimator_linear_3 = nn.Linear(128, 1)

        self.rate_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, trajectory_embedding, destination_zone_embedding, candidate_road_embedding, metric_distance, metric_angle):
        q = torch.cat([
            trajectory_embedding,
            destination_zone_embedding.unsqueeze(1).expand(-1, trajectory_embedding.size(1), -1)
        ], dim=-1).unsqueeze(2)
        k = torch.cat([candidate_road_embedding, self.metric_distance_proj(metric_distance.unsqueeze(-1)), self.metric_angle_proj(metric_angle.unsqueeze(-1))], dim=-1)

        q_projected = self.q_proj(q)
        k_projected = self.k_proj(k)
        logits = self.v_proj(torch.tanh(q_projected + k_projected)).squeeze(-1)
        
        # 计算rate：使用q和logits
        q_squeezed = q_projected.squeeze(2)  # [B, T, hidden_dim]
        rate_pred = self.rate_estimator(torch.cat([
            q_squeezed.unsqueeze(2).expand(-1, -1, candidate_road_embedding.size(2), -1),
            logits.unsqueeze(-1)
        ], dim=-1)).squeeze(-1)
        
        time_pred = self.time_estimator_linear_3(F.gelu(torch.cat([
            self.time_estimator_linear_1(trajectory_embedding.unsqueeze(2).expand(-1, -1, candidate_road_embedding.size(2), -1)),
            self.time_estimator_linear_2(candidate_road_embedding)
        ], dim=-1))).squeeze(3)

        return logits, time_pred, rate_pred

    # Compared to `forward()`, the batch size of the input here is 1, and only the last next spatio-temporal point is predicted
    @torch.no_grad()
    def infer(self, trajectory_embedding, destination_zone_embedding, candidate_road_embedding, metric_distance, metric_angle):
        q = torch.cat([
            trajectory_embedding[:, -1, :],
            destination_zone_embedding,
        ], dim=-1).unsqueeze(1)
        k = torch.cat([candidate_road_embedding, self.metric_distance_proj(metric_distance.unsqueeze(-1)), self.metric_angle_proj(metric_angle.unsqueeze(-1))], dim=-1)

        q_projected = self.q_proj(q)
        k_projected = self.k_proj(k)
        logits = self.v_proj(torch.tanh(q_projected + k_projected)).squeeze(-1)
        
        # 计算rate：使用q和logits
        q_squeezed = q_projected.squeeze(1)  # [B, hidden_dim]
        rate_pred = self.rate_estimator(torch.cat([
            q_squeezed.unsqueeze(1).expand(-1, candidate_road_embedding.size(1), -1),
            logits.unsqueeze(-1)
        ], dim=-1)).squeeze(-1)
        
        time_pred = self.time_estimator_linear_3(F.gelu(torch.cat([
            self.time_estimator_linear_1(trajectory_embedding[:, -1, :].unsqueeze(1).expand(-1, candidate_road_embedding.size(1), -1)),
            self.time_estimator_linear_2(candidate_road_embedding)
        ], dim=-1))).squeeze(2)

        return logits, time_pred, rate_pred
