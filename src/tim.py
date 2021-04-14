import torch.nn.functional as F
import torch.nn as nn
from .utils import get_mi, get_cond_entropy, get_loss, get_features, get_entropy, get_one_hot, warp_tqdm
import collections
from sacred import Ingredient
import torch
import time

import timeit
import gc
from src.losses import SupConLoss

from .utils import get_metric
import numpy as np
#from sklearn.cluster import KMeans

tim_ingredient = Ingredient('tim')
@tim_ingredient.config
def config():
    classifier = 'l2'
    temp = 15
    xent_weight = 0.1
    hy_weight = 1.0
    hyx_weight = 0.1
    loss_weights = [xent_weight, hy_weight, hyx_weight]  # [Xent, H(Y), H(Y|X)]
    lr = 1e-4
    iter = 150
    alpha = 1.0


class TIM(object):
    @tim_ingredient.capture
    def __init__(self, classifier, temp, loss_weights, iter, model, benchmark=False, disable_tqdm=False):
        self.classifier = classifier
        self.temp = temp
        self.loss_weights = loss_weights.copy()
        self.iter = iter
        self.model = model
        self.init_info_lists()
        self.disable_tqdm = disable_tqdm
        self.benchmark = benchmark

    def init_info_lists(self):
        self.timestamps = []
        self.mutual_infos = []
        self.entropy = []
        self.cond_entropy = []
        self.test_acc = []
        self.losses = []

    def get_logits(self, samples):
        n_tasks = samples.size(0)
        if self.classifier == 'cosine':
            logits = self.temp * F.cosine_similarity(samples[:, None, :], self.weights[None, :, :], dim=2)
        elif self.classifier == 'l2':
            logits = self.temp * (samples.matmul(self.weights.transpose(1, 2)) - 1 / 2 * (self.weights**2).sum(2).view(n_tasks, 1, -1) - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))  #
        return logits

    def get_preds(self, samples):
        logits = self.get_logits(samples)
        return logits.argmax(2)

    def proto_rect(self, gallery, query, meta_val_way=5, shot=1):
        eta = gallery.mean(0) - query.mean(0)
        query = query + eta[None, :]
        query_aug = torch.cat((gallery, query), axis=0)
        gallery_ = gallery.view(meta_val_way, shot, gallery.shape[1]).mean(1)

        distance = get_metric('cosine')(gallery_, query_aug)
        predict = torch.argmin(distance, dim=1)
        cos_sim = F.cosine_similarity(query_aug[:, None, :], gallery_[None, :, :], dim=2)
        cos_sim = self.temp * cos_sim
        W = F.softmax(cos_sim,dim=1)
        gallery_list = [(W[predict==i,i].unsqueeze(1)*query_aug[predict==i]).mean(0,keepdim=True) for i in predict.unique()]
        gallery = torch.cat(gallery_list,dim=0)
        return gallery

    def cluster(self, init, support, query, meta_val_way, shot):
        model = KMeans(n_clusters=meta_val_way, init=init, n_init=1, algorithm='auto')
        data = np.concatenate((support, query), axis=0)
        model.fit(data)
        centers = model.cluster_centers_
        return torch.from_numpy(centers)

    def init_weights(self, support, query, y_s, y_q, meta_val_way, shot, init_mode):
        self.model.eval()
        t0 = time.time()
        n_tasks = support.size(0)

        print("  ==> initialization with method: {}".format(init_mode))
        if init_mode == 'proto_rect':
            support_list = []
            tqdm_loop = warp_tqdm(list(range(n_tasks)),
                                  disable_tqdm=self.disable_tqdm)
            support_ = support.detach().cpu()
            query_ = query.detach().cpu()
            for i in tqdm_loop:
                foo = self.proto_rect(support_[i], query_[i], meta_val_way, shot)
                bar = foo[None, :, :]
                support_list.append(bar)
            self.weights = torch.cat(support_list, dim=0).to(support.device)


            if self.benchmark:
        
                stmt = """
support_list = []
support_ = support.detach().cpu()
query_ = query.detach().cpu()
foo = self.proto_rect(support_[0], query_[0], meta_val_way, shot)
bar = foo[None, :, :]
support_list.append(bar)
_ = torch.cat(support_list, dim=0).to(support.device)
torch.cuda.synchronize()
                """
                setup_stmt = """
gc.enable()
torch.cuda.synchronize()
                """
                globals()['self'], globals()['support'] = self, support
                globals()['query'], globals()['meta_val_way'] = query, meta_val_way
                globals()['shot'] = shot
                torch.cuda.synchronize()
                _ = timeit.timeit(stmt, setup=setup_stmt,
                        globals=globals(), number=1000)
        
                run_iter = 1000
                torch.cuda.synchronize()
                total_time = timeit.timeit(stmt, setup=setup_stmt,
                        globals=globals(), number=run_iter)
        
                print('proto_rect: {}'.format(total_time / run_iter))

        elif init_mode == 'cluster':

            support_list = []
            tqdm_loop = warp_tqdm(list(range(n_tasks)),
                                  disable_tqdm=self.disable_tqdm)
            support_ = support.detach().cpu()
            query_ = query.detach().cpu()
            for i in tqdm_loop:
                init_ = self.proto_rect(support_[i], query_[i], meta_val_way, shot)
                foo = self.cluster(init_.numpy(), support_[i].numpy(), query_[i].numpy(), meta_val_way, shot)
                bar = foo[None, :, :]
                support_list.append(bar)
            self.weights = torch.cat(support_list, dim=0).to(support.device)

        else:
            one_hot = get_one_hot(y_s)
            counts = one_hot.sum(1).view(n_tasks, -1, 1)
            weights = one_hot.transpose(1, 2).matmul(support)
            self.weights = weights / counts
        self.record_info(new_time=time.time()-t0,
                         support=support,
                         query=query,
                         y_s=y_s,
                         y_q=y_q)
        self.model.train()

    def compute_lambda(self, support, query, y_s):
        self.N_s, self.N_q = support.size(1), query.size(1)
        self.num_classes = torch.unique(y_s).size(0)
        if self.loss_weights[0] == 'auto':
            self.loss_weights[0] = (1 + self.loss_weights[2]) * self.N_s / self.N_q

    def record_info(self, new_time, support, query, y_s, y_q):
        logits_q = self.get_logits(query).detach()
        preds_q = logits_q.argmax(2)
        q_probs = logits_q.softmax(2)
        self.timestamps.append(new_time)
        self.mutual_infos.append(get_mi(probs=q_probs))
        self.entropy.append(get_entropy(probs=q_probs.detach()))
        self.cond_entropy.append(get_cond_entropy(probs=q_probs.detach()))
        self.test_acc.append((preds_q == y_q).float().mean(1, keepdim=True))

    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        self.cond_entropy = torch.cat(self.cond_entropy, dim=1).cpu().numpy()
        self.entropy = torch.cat(self.entropy, dim=1).cpu().numpy()
        self.mutual_infos = torch.cat(self.mutual_infos, dim=1).cpu().numpy()
        # self.losses = torch.cat(self.losses, dim=1)
        return {'timestamps': self.timestamps, 'mutual_info': self.mutual_infos,
                'entropy': self.entropy, 'cond_entropy': self.cond_entropy,
                'acc': self.test_acc, 'losses': self.losses}

    def run_adaptation(self, support, query, y_s, y_q, callback):
        pass


class TIM_GD(TIM):
    @tim_ingredient.capture
    def __init__(self, lr, model, benchmark=False, disable_tqdm=False):
        super().__init__(model=model, benchmark=benchmark, disable_tqdm=disable_tqdm)
        self.lr = lr

    def run_adaptation(self, support, query, y_s, y_q, callback):
        t0 = time.time()
        self.weights.requires_grad_()
        self.optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s)
        self.model.train()
        supcon = SupConLoss(temperature=0.1).to(support.device)
        tqdm_loop = warp_tqdm(list(range(self.iter)),
                              disable_tqdm=self.disable_tqdm)
        for i in tqdm_loop:
            logits_s = self.get_logits(support)
            logits_q = self.get_logits(query)

            ce = - (y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12)).sum(2).mean(1).sum(0)  # Taking the mean over samples within a task, and summing over all samples
            q_probs = logits_q.softmax(2)
            q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
            q_ent = - (q_probs.mean(1) * torch.log(q_probs.mean(1))).sum(1).sum(0)

            loss = self.loss_weights[0] * ce - (self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            t1 = time.time()
            self.model.eval()
            if callback is not None:
                P_q = self.get_logits(query).softmax(2).detach()
                prec = (P_q.argmax(2) == y_q).float().mean()
                callback.scalar('prec', i, prec, title='Precision')

            self.record_info(new_time=t1-t0,
                             support=support,
                             query=query,
                             y_s=y_s,
                             y_q=y_q)
            self.model.train()
            t0 = time.time()


class TIM_ADM(TIM):
    @tim_ingredient.capture
    def __init__(self, model, alpha, benchmark=False, disable_tqdm=False):
        super().__init__(model=model, benchmark=benchmark, disable_tqdm=disable_tqdm)
        self.alpha = alpha

    def q_update(self, P):

        l1, l2 = self.loss_weights[1], self.loss_weights[2]
        l3 = 1.0
        alpha = 1 + l2 / l3
        beta = l1 / (l1 + l3)

        # print(f"==> Alpha={alpha} \t Beta={beta}")
        Q = (P ** alpha) / ((P ** alpha).sum(dim=1, keepdim=True)) ** beta
        self.Q = (Q / Q.sum(dim=2, keepdim=True)).float()

    def weights_update(self, src_samples, qry_samples, W_support):
        n_tasks = src_samples.size(0)
        P_s = self.get_logits(src_samples).softmax(2)
        P_q = self.get_logits(qry_samples).softmax(2)
        src_part = self.loss_weights[0] / (1 + self.loss_weights[2]) * W_support.transpose(1, 2).matmul(src_samples)
        src_part += self.loss_weights[0] / (1 + self.loss_weights[2]) * (self.weights * P_s.sum(1, keepdim=True).transpose(1, 2) - P_s.transpose(1, 2).matmul(src_samples))
        src_norm = self.loss_weights[0] / (1 + self.loss_weights[2]) * W_support.sum(1).view(n_tasks, -1, 1)

        qry_part = self.N_s / self.N_q * self.Q.transpose(1, 2).matmul(qry_samples)
        qry_part += self.N_s / self.N_q * (self.weights * P_q.sum(1, keepdim=True).transpose(1, 2) - P_q.transpose(1, 2).matmul(qry_samples))
        qry_norm = self.N_s / self.N_q * self.Q.sum(1).view(n_tasks, -1, 1)

        new_weights = (src_part + qry_part) / (src_norm + qry_norm)
        self.weights = self.weights + self.alpha * (new_weights - self.weights)

    def run_adaptation(self, support, query, y_s, y_q, callback):
        tqdm_loop = warp_tqdm(list(range(self.iter)),
                              disable_tqdm=self.disable_tqdm)
        t0 = time.time()
        W_support = get_one_hot(y_s)
        for i in tqdm_loop:
            P_q = self.get_logits(query).softmax(2)
            self.q_update(P=P_q)
            self.weights_update(support, query, W_support)
            t1 = time.time()
            if callback is not None:
                callback.scalar('acc', i, self.test_acc[-1].mean(), title='Precision')
                callback.scalars(['cond_ent', 'marg_ent'], i, [self.cond_entropy[-1].mean(), self.entropy[-1].mean()], title='Entropies')
            self.record_info(new_time=t1-t0,
                             support=support,
                             query=query,
                             y_s=y_s,
                             y_q=y_q)
            t0 = time.time()

        if self.benchmark:
    
            stmt = """
W_support = get_one_hot(y_s)
for i in range(self.iter):
    P_q = self.get_logits(query).softmax(2)
    self.q_update(P=P_q)
    self.weights_update(support, query, W_support)
torch.cuda.synchronize()
            """
            setup_stmt = """
gc.enable()
torch.cuda.synchronize()
            """
            globals()['y_s'] = y_s
            globals()['self'], globals()['query'] = self, query
            globals()['support'], globals()['W_support'] = support, W_support
            torch.cuda.synchronize()
            _ = timeit.timeit(stmt, setup=setup_stmt,
                    globals=globals(), number=100)
    
            run_iter = 100
            torch.cuda.synchronize()
            total_time = timeit.timeit(stmt, setup=setup_stmt,
                    globals=globals(), number=run_iter)
    
            print('adaptation: {}'.format(total_time / run_iter))

