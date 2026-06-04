# copied from pl_bolts but modified due to an import error in pl_bolts
# see from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import math
import warnings
from typing import List, Optional

import numpy as np

import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from torch import inf

from fast_soft_sort.pytorch_ops import soft_rank
# from geomloss import SamplesLoss
from lightly.models.modules.memory_bank import MemoryBankModule

from haversine import haversine_vector


class BYOLLoss(nn.Module):
    """
    BYOL loss function as described in Grill et al. (NeurIPS 2020).
    Computes the mean squared error between L2-normalized predictions and targets.

    Given two views (p1, z2) and (p2, z1), the loss is symmetric:
        loss = MSE(normalize(p1), normalize(z2)) + MSE(normalize(p2), normalize(z1))
    """
    def __init__(self):
        super().__init__()

    def forward(self, p1: torch.Tensor, z2: torch.Tensor,
                      p2: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        # L2 normalize the vectors
        p1 = F.normalize(p1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        z1 = F.normalize(z1, dim=-1)

        # Compute MSE loss
        loss1 = 2 - 2 * (p1 * z2).sum(dim=-1).mean()
        loss2 = 2 - 2 * (p2 * z1).sum(dim=-1).mean()
        return 0.5 * (loss1 + loss2)


class SupervisedNTXentLoss(MemoryBankModule):
    def __init__(
        self,
        temperature: float = 0.1,
        min_dist_with_weight: float = 0.5,
        max_dist_with_weight: float = 1.0,
    ):
        super().__init__()
        # super().__init__(size=4092, gather_distributed=False)
        self.temperature = temperature
        self.min_dist_with_weight = min_dist_with_weight / temperature
        self.max_dist_with_weight = max_dist_with_weight / temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, gps_locations: torch.Tensor):
        device = out0.device
        batch_size, _ = out0.shape
        gps_locations = gps_locations.to(device)

        # normalize the output to length 1
        gps_locations = F.normalize(gps_locations, dim=1)
        out0 = F.normalize(out0, dim=1)
        
        out0_large = out0
        diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)

        # calculate similiarities
        # here n = batch_size and m = batch_size * world_size
        # the resulting vectors have shape (n, m)
        # & remove simliarities between same views of the same image
        logits = torch.einsum("nc,mc->nm", out0, out0_large) / self.temperature
        logits = logits[~diag_mask].view(batch_size, -1)

        locs_dists = torch.cdist(gps_locations, gps_locations) / self.temperature
        locs_dists = locs_dists[~diag_mask].view(batch_size, -1)
        locs_dists = locs_dists.to(torch.float32)

        # reverse distances to match similarties from features
        locs_dists = (1 / self.temperature) - locs_dists

        mask = (locs_dists >= self.min_dist_with_weight) & (locs_dists <= self.max_dist_with_weight)
        locs_dists_clipped = locs_dists - self.min_dist_with_weight
        locs_dists_clipped[~mask] = 0

        locs_dists_norm = locs_dists_clipped / locs_dists_clipped.sum(1).view(-1, 1)

        N, M = logits.shape
        # N, M = logits_00.shap  # only query
        
        losses = []
        for i in torch.arange(N):
            sample_logits = logits[i].repeat(M, 1).double()
            sample_labels = torch.arange(M, device=device, dtype=torch.long)
            weighted_cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean", weight=locs_dists_norm[i].double())
            loss_i = weighted_cross_entropy(sample_logits, sample_labels)
            losses.append(loss_i)

        loss = torch.mean(torch.stack(losses))
        
        return loss


class GeographyNTXentLoss(MemoryBankModule):
    def __init__(
        self,
        temperature: float = 0.1,
        min_dist_with_weight: float = 0.5,
        max_dist_with_weight: float = 1.0,
    ):
        super().__init__()
        # super().__init__(size=4092, gather_distributed=False)
        self.temperature = temperature
        self.min_dist_with_weight = min_dist_with_weight / temperature
        self.max_dist_with_weight = max_dist_with_weight / temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, gps_locations: torch.Tensor):
        device = out0.device
        batch_size, _ = out0.shape
        gps_locations = gps_locations.to(device)

        # normalize the output to length 1
        gps_locations = F.normalize(gps_locations, dim=1)
        out0 = F.normalize(out0, dim=1)
        
        out0_large = out0
        diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)

        # calculate similiarities
        # here n = batch_size and m = batch_size * world_size
        # the resulting vectors have shape (n, m)
        # & remove simliarities between same views of the same image
        logits = torch.einsum("nc,mc->nm", out0, out0_large) / self.temperature
        logits = logits[~diag_mask].view(batch_size, -1)

        locs_dists = torch.cdist(gps_locations, gps_locations) / self.temperature
        locs_dists = locs_dists[~diag_mask].view(batch_size, -1)
        locs_dists = locs_dists.to(torch.float32)

        # reverse distances to match similarties from features
        locs_dists = (1 / self.temperature) - locs_dists

        mask = (locs_dists >= self.min_dist_with_weight) & (locs_dists <= self.max_dist_with_weight)
        locs_dists_clipped = locs_dists - self.min_dist_with_weight
        locs_dists_clipped[~mask] = 0

        locs_dists_norm = locs_dists_clipped / locs_dists_clipped.sum(1).view(-1, 1)

        loss = self.cross_entropy(logits, locs_dists_norm)

        return loss


class GeographyMSELoss(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        spacing: float = 0.1,
        min_dist_with_weight: float = 0.5,
        max_dist_with_weight: float = 1.0,
        soft_margin: float = 0.01,
    ):
        super().__init__()
        # super().__init__(size=4092, gather_distributed=False)
        self.temperature = temperature
        self.spacing = spacing / temperature
        # self.min_dist_with_weight = min_dist_with_weight / temperature
        self.max_dist_with_weight = max_dist_with_weight / temperature
        self.soft_margin = soft_margin / (temperature ** 2)  # be careful: only if mse
        self.mse = torch.nn.MSELoss(reduction="none")

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, gps_locations: torch.Tensor):
        device = out0.device
        batch_size, _ = out0.shape

        # normalize the output to length 1
        out0 = F.normalize(out0, dim=1)
        
        out0_large = out0
        diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)

        # calculate similiarities
        # here n = batch_size and m = batch_size * world_size
        # the resulting vectors have shape (n, m)
        # & remove simliarities between same views of the same image
        logits = torch.einsum("nc,mc->nm", out0, out0_large) / self.temperature
        logits = logits[~diag_mask].view(batch_size, -1)

        gps_locations_np = gps_locations.numpy()[:, [1, 0]]
        locs_dists = torch.tensor(haversine_vector(gps_locations_np, gps_locations_np, comb=True))
        locs_dists = locs_dists.to(device)
        locs_dists = (locs_dists / 8000) / self.temperature
        locs_dists = locs_dists[~diag_mask].view(batch_size, -1)
        locs_dists = locs_dists.to(torch.float32)

        # reverse similarties to match distances from locations
        # logits = ((1 / self.temperature) - self.spacing) - logits
        logits = ((1 / self.temperature) - logits) - self.spacing

        losses = self.mse(logits, locs_dists)
        
        mask1 = locs_dists <= self.max_dist_with_weight
        weighted_losses = losses[mask1]

        mask2 = weighted_losses <= self.soft_margin
        weighted_losses[mask2] = 0
    
        return weighted_losses.mean()


class SoftRankLoss(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        reg_strength: float = 0.001,
        min_dist_km: int = 0,
        max_dist_km: int = 3000,
        soft_margin: int = 100,
        loss_type: str = 'mse',
        distance_measure: str = 'haversine'
    ):
        super().__init__()
        # super().__init__(size=4092, gather_distributed=False)
        self.temperature = temperature
        self.reg_strength = reg_strength
        self.min_dist_km = min_dist_km
        self.max_dist_km = max_dist_km
        self.soft_margin = soft_margin
        self.loss_type = loss_type
        self.distance_measure = distance_measure

        if self.loss_type == 'mse':
            self.loss = torch.nn.MSELoss(reduction="none")
        elif self.loss_type == 'l1':
            self.loss = torch.nn.L1Loss(reduction="none")

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, gps_locations: torch.Tensor):
        device = out0.device
        batch_size, _ = out0.shape

        # normalize the output to length 1
        out0 = F.normalize(out0, dim=1)

        out0_large = out0
        diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)

        # calculate similiarities
        # here n = batch_size and m = batch_size * world_size
        # the resulting vectors have shape (n, m)
        # & remove simliarities between same views of the same image
        logits = torch.einsum("nc,mc->nm", out0, out0_large) / self.temperature
        logits = logits[~diag_mask].view(batch_size, -1)

        if self.distance_measure == 'haversine':
            gps_locations_np = gps_locations.numpy()  # * 0.5
            locs_dists = torch.tensor(haversine_vector(gps_locations_np, gps_locations_np, comb=True))
        elif self.distance_measure == 'euclidean':
            locs_dists = torch.cdist(gps_locations, gps_locations, p=2)

        locs_dists = locs_dists.to(device)
        locs_dists = locs_dists[~diag_mask].view(batch_size, -1)
        locs_dists = locs_dists.to(torch.float32)

        mask1 = (locs_dists <= self.max_dist_km) & (locs_dists >= self.min_dist_km)
        # to match similarity of embeddings (high == close)
        # locs_dists_inv = locs_dists.max() - locs_dists

        scaling_factor = torch.arange(1, batch_size, dtype=torch.double).norm()
        loss_margin = (self.soft_margin / scaling_factor) ** 2
        if self.loss_type == 'l1':
            loss_margin = torch.sqrt(loss_margin)

        x_softranks = batch_size - soft_rank(logits.cpu(), regularization_strength=self.reg_strength).to(device)
        x_softranks = x_softranks / x_softranks.norm(dim=1, keepdim=True)
        x_softranks = x_softranks.double()

        y_hardranks = torch.argsort(torch.argsort(locs_dists, axis=1), axis=1) + 1
        y_hardranks = y_hardranks.double()
        y_hardranks = y_hardranks / y_hardranks.norm(dim=1, keepdim=True)

        # loss for distances inside max_km_radius
        losses = self.loss(x_softranks, y_hardranks)
        weighted_losses = losses[mask1]

        mask2 = weighted_losses <= loss_margin
        weighted_losses[mask2] = 0

        return weighted_losses.mean() * scaling_factor
    
        # loss for distances outside max_km_radius
        # max_rank = torch.count_nonzero(locs_dists > self.max_dist_km, axis=1)
        # max_rank_norm = max_rank / scaling_factor
        # x_softranks_clipped = x_softranks.clip(max=max_rank_norm[:, None])
        # y_hardranks_clipped = y_hardranks.clip(max=max_rank_norm[:, None])

        # losses = self.loss(x_softranks_clipped, y_hardranks_clipped)
        # weighted_losses_distant = losses[~mask1]
    
        # return weighted_losses.mean() * scaling_factor + weighted_losses_distant.mean() * scaling_factor


class SoftRankL2Loss(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        reg_strength: float = 0.001,
        min_dist_km: int = 0,
        max_dist_km: int = 3000,
        soft_margin: int = 100,
        loss_type: str = 'mse'
    ):
        super().__init__()
        # super().__init__(size=4092, gather_distributed=False)
        self.temperature = temperature
        self.reg_strength = reg_strength
        self.min_dist_km = min_dist_km
        self.max_dist_km = max_dist_km
        self.soft_margin = soft_margin
        self.loss_type = loss_type

        if self.loss_type == 'mse':
            self.loss = torch.nn.MSELoss(reduction="none")
        elif self.loss_type == 'l1':
            self.loss = torch.nn.L1Loss(reduction="none")

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, gps_locations: torch.Tensor):
        device = out0.device
        batch_size, _ = out0.shape
        gps_locations_l2 = gps_locations.to(device)

        # normalize the output to length 1
        out0 = F.normalize(out0, dim=1)
        gps_locations_l2 = F.normalize(gps_locations_l2, dim=1)

        out0_large = out0
        diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)

        gps_locations_np = gps_locations.numpy()  # * 0.5
        locs_dists_hav = torch.tensor(haversine_vector(gps_locations_np, gps_locations_np, comb=True))
        locs_dists_hav = locs_dists_hav.to(device)
        locs_dists_hav = locs_dists_hav[~diag_mask].view(batch_size, -1)
        locs_dists_hav = locs_dists_hav.to(torch.float32)

        # calculate similiarities
        # here n = batch_size and m = batch_size * world_size
        # the resulting vectors have shape (n, m)
        # & remove simliarities between same views of the same image
        logits = torch.einsum("nc,mc->nm", out0, out0_large) / self.temperature
        logits = logits[~diag_mask].view(batch_size, -1)

        locs_dists = torch.cdist(gps_locations_l2, gps_locations_l2)
        locs_dists = locs_dists[~diag_mask].view(batch_size, -1)
        locs_dists = locs_dists.to(torch.float32)

        mask1 = (locs_dists_hav <= self.max_dist_km) & (locs_dists_hav >= self.min_dist_km)
        # to match similarity of embeddings (high == close)
        # locs_dists_inv = locs_dists.max() - locs_dists

        scaling_factor = torch.arange(1, batch_size, dtype=torch.double).norm()
        loss_margin = (self.soft_margin / scaling_factor) ** 2
        if self.loss_type == 'l1':
            loss_margin = torch.sqrt(loss_margin)

        x_softranks = batch_size - soft_rank(logits.cpu(), regularization_strength=self.reg_strength).to(device)
        x_softranks = x_softranks / x_softranks.norm(dim=1, keepdim=True)
        x_softranks = x_softranks.double()

        y_hardranks = torch.argsort(torch.argsort(locs_dists, axis=1), axis=1) + 1
        y_hardranks = y_hardranks.double()
        y_hardranks = y_hardranks / y_hardranks.norm(dim=1, keepdim=True)

        # loss for distances inside max_km_radius
        losses = self.loss(x_softranks, y_hardranks)
        weighted_losses = losses[mask1]

        mask2 = weighted_losses <= loss_margin
        weighted_losses[mask2] = 0

        return weighted_losses.mean() * scaling_factor


# class SoftRankLoss(torch.nn.Module):
#     def __init__(
#         self,
#         temperature: float = 0.1,
#         reg_strength: float = 0.001,
#         min_dist_km: int = 0,
#         max_dist_km: int = 3000,
#         soft_margin: int = 100,
#         loss_type: str = 'mse',
#     ):
#         super().__init__()
#         # super().__init__(size=4092, gather_distributed=False)
#         self.temperature = temperature
#         self.reg_strength = reg_strength
#         self.min_dist_with_weight = 0.5 / temperature
#         self.max_dist_with_weight = 1.0 / temperature
#         self.soft_margin = 0.001
#         self.mse = torch.nn.MSELoss(reduction="none")

#     def forward(self, out0: torch.Tensor, out1: torch.Tensor, gps_locations: torch.Tensor):
#         device = out0.device
#         batch_size, _ = out0.shape
#         # gps_locations = gps_locations.to(device)

#         # normalize the output to length 1
#         # gps_locations = F.normalize(gps_locations, dim=1)
#         out0 = F.normalize(out0, dim=1)

#         out0_large = out0
#         diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)

#         # calculate similiarities
#         # here n = batch_size and m = batch_size * world_size
#         # the resulting vectors have shape (n, m)
#         # & remove simliarities between same views of the same image
#         logits = torch.einsum("nc,mc->nm", out0, out0_large) / self.temperature
#         logits = logits[~diag_mask].view(batch_size, -1)

#         gps_np_swapped = gps_locations.numpy() * 0.5
#         locs_dists = torch.tensor(haversine_vector(gps_np_swapped, gps_np_swapped, comb=True))
#         locs_dists = locs_dists.to(device)
#         # locs_dists = torch.cdist(gps_locations, gps_locations) / self.temperature
#         locs_dists = locs_dists[~diag_mask].view(batch_size, -1)
#         locs_dists = locs_dists.to(torch.float32)

#         # reverse distances to match similarties from features
#         # locs_dists = (1 / self.temperature) - locs_dists
#         locs_dists = locs_dists.max() - locs_dists

#         x_softranks = soft_rank(logits.cpu(), regularization_strength=self.reg_strength).to(device)
#         x_softranks = x_softranks / x_softranks.norm(dim=1, keepdim=True)
#         x_softranks = x_softranks.double()
#         # y_softranks = soft_rank(locs_dists.cpu(), regularization_strength=self.reg_strength).to(device)

#         y_hardranks = torch.argsort(torch.argsort(locs_dists, axis=1), axis=1) + 1
#         y_hardranks = y_hardranks.double()
#         y_hardranks = y_hardranks / y_hardranks.norm(dim=1, keepdim=True)

#         losses = self.mse(x_softranks, y_hardranks)
        
#         mask1 = (locs_dists >= (locs_dists.max() - 3000))
#         weighted_losses = losses[mask1]

#         mask2 = weighted_losses <= self.soft_margin
#         weighted_losses[mask2] = 0
    
#         return weighted_losses.sum()


# class GeographyConsistencyLoss(torch.nn.Module):
#     def __init__(
#         self,
#         algorithm: str = 'rank_approximation',
#         reach: int = 400,
#         loss_batch_size: int = 64,
#         reg_strength: float = 0.00001,
#         lower_bound: Optional[float] = None,
#         upper_bound: Optional[float] = None,
#         # reg_strength: float = 0.000003,
#         pct_threshold: Optional[float] = None,
#         p1_weighting: Optional[int] = None,
#         p2_weighting: Optional[int] = None,
#         p3_weighting: Optional[int] = None
#     ):
#         super(GeographyConsistencyLoss, self).__init__()
#         self.algorithm = algorithm
#         self.reach = reach
#         self.loss_batch_size = loss_batch_size
#         self.lower_bound = lower_bound
#         self.upper_bound = upper_bound
#         self.reg_strength = reg_strength
#         self.pct_threshold = pct_threshold
#         self.p1_weighting = p1_weighting
#         self.p2_weighting = p2_weighting
#         self.p3_weighting = p3_weighting
#         self.tripletloss = torch.nn.TripletMarginLoss(0)
#         # self.geomloss = SamplesLoss(loss="sinkhorn", p=2, reach=self.reach)

#     def threshold_to_zero(self, x, batch_size, device):
#         threshold = (batch_size * batch_size / 2 - batch_size) * self.pct_threshold
#         return torch.where(torch.abs(x) <= threshold, torch.tensor(0., device=device), x)

#     def geographic_distance_weight(self, dist, device):
#         weight = torch.zeros_like(dist, device=device)
#         mask1 = (dist < self.p1_weighting)
#         mask2 = (dist >= self.p1_weighting) & (dist <= self.p2_weighting)
#         mask3 = (dist > self.p2_weighting) & (dist <= self.p3_weighting)
        
#         weight[mask1] = dist[mask1] / self.p1_weighting
#         weight[mask2] = 1
#         weight[mask3] = (self.p3_weighting - dist[mask3]) / (self.p3_weighting - self.p2_weighting)
        
#         return weight

#     def compute_weights(self, locs_flatten, device):
#         if self.p1_weighting is None:
#             return torch.ones_like(locs_flatten, device=device)
#         else:
#             return self.geographic_distance_weight(locs_flatten, device)

#     def kernel_density_estimation(self, distances: torch.Tensor, bandwidth: float = 1.0) -> torch.Tensor:
#         # Apply Gaussian kernel to distances
#         probs = torch.exp(-0.5 * (distances / bandwidth) ** 2)
        
#         # Normalize probabilities to sum to 1 along each row
#         probs = probs / torch.sum(probs)
        
#         return probs

#     def calculate_similarity_matrix(self, a, b, eps=1e-8):
#         """
#         added eps for numerical stability
#         """
#         a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
#         a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
#         b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
#         sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
#         return sim_mt

#     def forward(self, out0: torch.Tensor, gps_locations: torch.Tensor) -> float:
#         device = out0.device
#         batch_size, _ = out0.shape
#         gps_locations = gps_locations.to(device)

#         # compute distance between embeddings and between locations
#         embds_dists = torch.cdist(out0, out0)
#         locs_dists = torch.cdist(gps_locations, gps_locations)

#         if self.algorithm == 'optimal_transport':
#             # mini-batch computation of loss to keep loss and gradients within reasonable range
#             batch_loss_loss = []
            
#             for i in torch.arange(0, batch_size, self.loss_batch_size):

#                 # subset distance matrix selection into loss batches
#                 full_triangl_sel = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
#                 full_triangl_sel[i:i + self.loss_batch_size, i:i + self.loss_batch_size] = True
                
#                 batch_triangl_sel = torch.tril(torch.ones((batch_size, batch_size), dtype=torch.bool, device=device), -1)
#                 triangl_sel = torch.logical_and(full_triangl_sel, batch_triangl_sel)
                
#                 embds_flatten = embds_dists[triangl_sel]
#                 locs_flatten = locs_dists[triangl_sel]

#                 # compute kernel density estimation 
#                 embds_density = self.kernel_density_estimation(embds_flatten)
#                 locs_density = self.kernel_density_estimation(locs_flatten)
        
#                 # compute loc-weights
#                 weights = self.compute_weights(locs_flatten, device)
            
#                 # expand to 2d array and convert to double for SamplesLoss requirements 
#                 x1 = embds_density.unsqueeze(0).double()
#                 x2 = locs_density.unsqueeze(0).double()
#                 batch_loss_loss.append(self.geomloss(x1, x2))

#             loss = torch.sum(torch.stack(batch_loss_loss))

#         elif self.algorithm == 'rank_approximation':
#             # flatten embedding and location distances
#             triangl_sel = torch.tril(torch.ones((batch_size, batch_size), dtype=torch.bool, device=device), -1)
#             embds_flatten = embds_dists[triangl_sel]
#             locs_flatten = locs_dists[triangl_sel]

#             # compute loc-weights
#             weights = self.compute_weights(locs_flatten, device)

#             embds_expand = embds_flatten.unsqueeze(0).cpu()
#             locs_expand = locs_flatten.unsqueeze(0).cpu()

#             x_softranks = soft_rank(embds_expand, regularization_strength=self.reg_strength).to(device)
#             y_softranks = soft_rank(locs_expand, regularization_strength=self.reg_strength).to(device)

#             y_hardranks = torch.argsort(torch.argsort(locs_flatten)) + 1
#             y_hardranks = y_hardranks.double()
#             y_hardranks = y_hardranks / y_hardranks.norm()

#             # x_softranks = x_softranks - x_softranks.mean()
#             # y_softranks = y_softranks - y_softranks.mean()
#             x_softranks = x_softranks / x_softranks.norm()
#             y_softranks = y_softranks / y_softranks.norm()
#             loss = 1 - (y_softranks * x_softranks).sum()
#             # loss = ((y_hardranks - x_softranks) ** 2).sum()

#         elif self.algorithm == 'triplet':
#             device = out0.device
#             batch_size, _ = out0.shape
#             gps_locations = gps_locations.to(device)

#             # compute distance between embeddings and between locations
#             sim_matrix = self.calculate_similarity_matrix(out0, out0)
#             gps_dists = torch.cdist(gps_locations, gps_locations)
#             triangl_sel = torch.tril(torch.ones((3, 3), dtype=torch.bool, device=device), -1)

#             if self.lower_bound is not None:
#                 sim_matrix = torch.clip(sim_matrix, self.lower_bound, self.upper_bound)
#                 sim_matrix = (sim_matrix - self.lower_bound) / (self.upper_bound - self.lower_bound)
                           
#             sim_matrix_exp = torch.exp(sim_matrix)

#             losses = []
#             for i in torch.arange(0, batch_size - 3, 3):
#                 sims_batch_flatten = sim_matrix_exp[i:i + 3, i:i + 3][triangl_sel]
#                 gps_batch_flatten = gps_dists[i:i + 3, i:i + 3][triangl_sel]
#                 loss_batch = -(torch.log(sims_batch_flatten[torch.argmin(gps_batch_flatten)] / sims_batch_flatten.sum()).sum())
#                 losses.append(loss_batch)
                
#             loss = torch.mean(torch.stack(losses))

#         elif self.algorithm == 'triplet2':
#             device = out0.device
#             batch_size, _ = out0.shape
#             gps_locations = gps_locations.to(device)

#             # compute distance between embeddings and between locations
#             gps_dists = torch.cdist(gps_locations, gps_locations)

#             indices = []
#             triangl_sel = torch.tril(torch.ones((3, 3), dtype=torch.bool), -1)

#             for i in torch.arange(0, 512 - 3, 3):
#                 gps_batch_flatten = gps_dists[i:i + 3, i:i + 3][triangl_sel]
#                 indices.append(torch.argsort(gps_batch_flatten) + i)

#             a_pos_neg_split = out0[torch.stack(indices)]

#             anchors = a_pos_neg_split[:, 0]
#             positives = a_pos_neg_split[:, 1]
#             negatives = a_pos_neg_split[:, 2]

#             loss = self.tripletloss(anchors, positives, negatives)

#         return loss


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr
    and base_lr followed by a cosine annealing schedule between base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " 
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim, grid_size, res, cls_token=False, device="cpu"
):
    """
    grid_size: int of the grid height and width
    res: array of size n, representing the resolution of a pixel (say, in meters),
    return:
    pos_embed: [n,grid_size*grid_size, embed_dim] or [n,1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # res = torch.FloatTensor(res).to(device)
    res = res.to(dtype=torch.float32, device=device)
    grid_h = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float32, device=device)
    grid = torch.meshgrid(
        grid_w, grid_h, indexing="xy"
    )  # here h goes first,direction reversed for numpy
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    # grid = grid.reshape([2, 1, grid_size, grid_size])
    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(
        embed_dim, grid
    )  #  # (nxH*W, D/2)
    pos_embed = pos_embed.float()
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [
                torch.zeros(
                    [n, 1, embed_dim], dtype=torch.float32, device=pos_embed.device
                ),
                pos_embed,
            ],
            dim=1,
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        try:
            num_patches = model.patch_embed.num_patches
        except AttributeError as err:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
