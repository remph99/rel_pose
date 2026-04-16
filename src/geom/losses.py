# import torch

# def geodesic_loss(Ps, Gs, train_val='train'):
#     """ Loss function for training network """

#     ii, jj = torch.tensor([0, 1]), torch.tensor([1, 0]) 

#     dP = Ps[:,jj] * Ps[:,ii].inv()
#     dG = Gs[0][:,jj] * Gs[0][:,ii].inv()
#     d = (dG * dP.inv()).log()

#     tau, phi = d.split([3,3], dim=-1) 
#     geodesic_loss_tr = tau.norm(dim=-1).mean()
#     geodesic_loss_rot = phi.norm(dim=-1).mean()

#     metrics = {
#         train_val+'_geo_loss_tr': (geodesic_loss_tr).detach().item(),
#         train_val+'_geo_loss_rot': (geodesic_loss_rot).detach().item(),
#     }

#     return geodesic_loss_tr, geodesic_loss_rot, metrics

##  修改平移loss   由平移残差范数修改为平移方向的夹角（弧度）    旋转loss保持不变，仍然保持geo_loss
import torch

def geodesic_loss(Ps, Gs, train_val='train'):
    """ Translation uses angular loss; rotation keeps geodesic loss. """

    ii, jj = torch.tensor([0, 1]), torch.tensor([1, 0])

    # relative pose: GT and prediction
    dP = Ps[:, jj] * Ps[:, ii].inv()
    dG = Gs[0][:, jj] * Gs[0][:, ii].inv()

    # ---------- translation angular loss ----------
    # SE3 data layout in this repo: [tx, ty, tz, qx, qy, qz, qw]
    t_gt = dP.data[..., :3]
    t_pr = dG.data[..., :3]

    eps = 1e-8
    min_norm = 1e-6

    n_gt = t_gt.norm(dim=-1)
    n_pr = t_pr.norm(dim=-1)
    valid = (n_gt > min_norm) & (n_pr > min_norm)

    cos_sim = (t_gt * t_pr).sum(dim=-1) / (n_gt * n_pr + eps)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
    tr_angle = torch.acos(cos_sim)  # radians in [0, pi]

    if valid.any():
        geodesic_loss_tr = tr_angle[valid].mean()
    else:
        geodesic_loss_tr = torch.zeros([], device=t_gt.device, dtype=t_gt.dtype)

    # ---------- rotation geodesic loss (unchanged) ----------
    d = (dG * dP.inv()).log()
    _, phi = d.split([3, 3], dim=-1)
    geodesic_loss_rot = phi.norm(dim=-1).mean()

    metrics = {
        train_val + '_geo_loss_tr': geodesic_loss_tr.detach().item(),
        train_val + '_geo_loss_rot': geodesic_loss_rot.detach().item(),
    }

    return geodesic_loss_tr, geodesic_loss_rot, metrics

