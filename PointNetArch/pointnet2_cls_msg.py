import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256) 
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, 1024)

        global_features = x
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l3_points, global_features

class get_model1(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(4096, [0.05, 0.1, 0.2], [16, 32, 64], in_channel,[[16, 32, 64], [32, 64, 64], [32, 64, 64]])
        self.sa2 = PointNetSetAbstractionMsg(1024, [0.1, 0.2, 0.4], [16, 32, 128], 192,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa4 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        x = l4_points.view(B, 1024)

        global_features = x
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l4_points, global_features


class get_loss(nn.Module):
    def __init__(self, class_weights = None, smoothing = 0.1):
        super(get_loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(
            weight=None,
            label_smoothing=smoothing
        )

    def forward(self, pred, target, trans_feat = None):
        loss = self.criterion(pred, target)

        #mat_diff = torch.matmul(trans_feat, trans_feat.transpose(2,1))
        #identity = torch.eye(trans_feat.shape[1]).to(pred.device)
        #reg_loss = torch.mean(torch.norm(identity - mat_diff, dim = (2,1)))

        total_loss = loss

        return total_loss


class hierarchical_loss(nn.Module):
    def __init__(self, species_to_genus_matrix, alpha = 0.5):
        super().__init__()
        self.register_buffer('M', species_to_genus_matrix)
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets_species, target_genus):
        loss_species = self.ce_loss(logits, targets_species)

        p_species = F.softmax(logits, dim = 2)

        p_genus = torch.matmul(p_species, self.M)

        log_p_genus = torch.log(p_genus + 1e-8)

        loss_genus = F.nll_loss(log_p_genus, target_genus)

        return loss_species + (self.alpha * loss_genus)