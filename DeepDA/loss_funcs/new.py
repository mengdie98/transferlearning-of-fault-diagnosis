from loss_funcs.adv import *
from loss_funcs.mmd import MMDLoss
import torch
import numpy as np
import torch.nn as nn


class DAAN_LMMD_Loss(AdversarialLoss, LambdaSheduler, MMDLoss):
    def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None,
                 gamma=1.0, max_iter=1000, x=0.5, **kwargs):
        super(DAAN_LMMD_Loss, self).__init__(gamma=gamma, max_iter=max_iter, **kwargs)
        super(MMDLoss, self).__init__(gamma, max_iter, **kwargs)
        super(DAAN_LMMD_Loss, self).__init__(kernel_type, kernel_mul, kernel_num, fix_sigma, **kwargs)
        super(AdversarialLoss, self).__init__(gamma=gamma, max_iter=max_iter, **kwargs)
        # MMDLoss.__init__(self, kernel_type=kernel_type, kernel_mul=kernel_mul, kernel_num=kernel_num,
        #                  fix_sigma=fix_sigma, **kwargs)
        # self.domain_classifier = Discriminator()
        self.num_class = num_class
        self.x = x
        self.local_classifiers = torch.nn.ModuleList()
        for _ in range(num_class):
            self.local_classifiers.append(Discriminator())

        self.d_g, self.d_l = 0, 0
        self.dynamic_factor = 0.5

    def forward(self, source, target, source_label, target_logits, source_logits):
        lamb = self.lamb()
        self.step()
        source_loss_g = self.get_adversarial_result(source, True, lamb)
        target_loss_g = self.get_adversarial_result(target, False, lamb)
        source_loss_l = self.get_local_adversarial_result(source, source_logits, True, lamb)
        target_loss_l = self.get_local_adversarial_result(target, target_logits, False, lamb)
        global_loss = 0.5 * (source_loss_g + target_loss_g) * 0.05
        local_loss = 0.5 * (source_loss_l + target_loss_l) * 0.01

        self.d_g = self.d_g + 2 * (1 - 2 * global_loss.cpu().item())
        self.d_l = self.d_l + 2 * (1 - 2 * (local_loss / self.num_class).cpu().item())

        adv_loss = (1 - self.dynamic_factor) * global_loss + self.dynamic_factor * local_loss

        # Add LMMD loss
        mmd_loss = self.cal_lmmd(source, target, source_label, target_logits)
        total_loss = adv_loss + self.x * mmd_loss
        return total_loss

    def get_local_adversarial_result(self, x, logits, c, source=True, lamb=1.0):
        loss_fn = torch.nn.BCELoss()
        x = ReverseLayerF.apply(x, lamb)
        loss_adv = 0.0

        for c in range(self.num_class):
            logits_c = logits[:, c].reshape((logits.shape[0], 1))  # (B, 1)
            features_c = logits_c * x
            domain_pred = self.local_classifiers[c](features_c)
            device = domain_pred.device
            if source:
                domain_label = torch.ones(len(x), 1).long()
            else:
                domain_label = torch.zeros(len(x), 1).long()
            loss_adv = loss_adv + loss_fn(domain_pred, domain_label.float().to(device))
        return loss_adv

    def update_dynamic_factor(self, epoch_length):
        if self.d_g == 0 and self.d_l == 0:
            self.dynamic_factor = 0.5
        else:
            self.d_g = self.d_g / epoch_length
            self.d_l = self.d_l / epoch_length
            self.dynamic_factor = 1 - self.d_g / (self.d_g + self.d_l)
        self.d_g, self.d_l = 0, 0

    def cal_lmmd(self, source, target, source_label, target_logits):
        if self.kernel_type == 'linear':
            raise NotImplementedError("Linear kernel is not supported yet.")
        elif self.kernel_type == 'rbf':
            batch_size = source.size()[0]
            weight_ss, weight_tt, weight_st = self.cal_weight(source_label, target_logits)
            weight_ss = torch.from_numpy(weight_ss).cuda()  # B, B
            weight_tt = torch.from_numpy(weight_tt).cuda()
            weight_st = torch.from_numpy(weight_st).cuda()

            kernels = self.gaussian_kernel(source, target,
                                           kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                           fix_sigma=self.fix_sigma)
            loss = torch.Tensor([0]).cuda()
            if torch.sum(torch.isnan(sum(kernels))):
                return loss
            SS = kernels[:batch_size, :batch_size]
            TT = kernels[batch_size:, batch_size:]
            ST = kernels[:batch_size, batch_size:]

            loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
            # Dynamic weighting
            lamb = self.lamb()
            self.step()
            loss = loss * lamb
            return loss

    def cal_weight(self, source_label, target_logits):
        batch_size = source_label.size()[0]
        source_label = source_label.cpu().data.numpy()
        source_label_onehot = np.eye(self.num_class)[source_label]  # one hot

        source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, self.num_class)
        source_label_sum[source_label_sum == 0] = 100
        source_label_onehot = source_label_onehot / source_label_sum  # label ratio

        # Pseudo label
        target_label = target_logits.cpu().data.max(1)[1].numpy()

        target_logits = target_logits.cpu().data.numpy()
        target_logits_sum = np.sum(target_logits, axis=0).reshape(1, self.num_class)
        target_logits_sum[target_logits_sum == 0] = 100
        target_logits = target_logits / target_logits_sum

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(source_label)
        set_t = set(target_label)
        count = 0
        for i in range(self.num_class):  # (B, C)
            if i in set_s and i in set_t:
                s_tvec = source_label_onehot[:, i].reshape(batch_size, -1)  # (B, 1)
                t_tvec = target_logits[:, i].reshape(batch_size, -1)  # (B, 1)

                ss = np.dot(s_tvec, s_tvec.T)  # (B, B)
                weight_ss = weight_ss + ss
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st
                count += 1

        length = count
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


# class Discriminator(torch.nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.fc1 = torch.nn.Linear(2048, 1024)
#         self.bn1 = torch.nn.BatchNorm1d(1024)
#         self.fc2 = torch.nn.Linear(1024, 1)
#         self.relu = torch.nn.ReLU(inplace=True)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return x


# class ReverseLayerF(torch.nn.Module):
#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.alpha = alpha
#         return x

#     @staticmethod
#     def backward(ctx, grad_output):
#         output = grad_output.neg() * ctx.alpha
#         return output, None

# class Discriminator(nn.Module):
#     def __init__(self, input_dim=256, hidden_dim=256):
#         super(Discriminator, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         layers = [
#             nn.Linear(input_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         ]
#         self.layers = torch.nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.layers(x)