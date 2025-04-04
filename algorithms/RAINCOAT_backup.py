import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.loss import SinkhornDistance
from pytorch_metric_learning import losses
from models.models import ResClassifier_MME, classifier

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def update(self, *args, **kwargs):
        raise NotImplementedError

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, fl=128):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.fl = fl
        
        # РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ РІРµСЃРѕРІ СЃ СЂР°Р·РјРµСЂРЅРѕСЃС‚СЊСЋ [in_channels, out_channels, modes1]
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        
        # РЎР»РѕРё РЅРѕСЂРјР°Р»РёР·Р°С†РёРё РґР»СЏ out_channels РєР°РЅР°Р»РѕРІ
        self.bn_r = nn.BatchNorm1d(out_channels)
        self.bn_p = nn.BatchNorm1d(out_channels)
        
        # РџСЂРѕРµРєС†РёРѕРЅРЅС‹Рµ СЃР»РѕРё СЃ out_channels РєР°РЅР°Р»Р°РјРё
        self.projection_r = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        self.projection_p = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.Tanh()
        )
        
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        
    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # РџСЂРёРјРµРЅСЏРµРј РѕРєРЅРѕ РҐРµРЅРЅРёРЅРіР° РґР»СЏ СѓРумРµРЅСЊС€РµРЅРёСЏ СѓС‚РµС‡РєРё СЃРїРµРєС‚СЂР°
        window = torch.hann_window(x.size(-1), device=x.device)
        x = x * window
        
        # Р'С‡РёСЃР»СЏРµРј FFT
        x_ft = torch.fft.rfft(x, norm='ortho')
        
        # РџСЂРёРјРµРЅСЏРµРј РґРІР° РЅР°Р±РѕСЂР° РІРµСЃРѕРІ РґР»СЏ Р»СѓС‡С€РµРіРѕ РїСЂРµРґСЃС‚Р°РІР»РµРЅРёСЏ С‡Р°СЃС‚РѕС‚РЅС‹С… РїСЂРёР·РЅР°РєРѕРІ
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        out_ft[:, :, :self.modes1] += self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights2)
        
        # РЅРѕСЂРјР°Р»РёР·СѓРµРј Рё РїСЂРёРјРµРЅСЏРµРј РЅРµР»РёРЅРµР№РЅРѕСЃС‚Рё
        r = out_ft[:, :, :self.modes1].abs()
        p = out_ft[:, :, :self.modes1].angle()
        
        # РќРѕСЂРјР°Р»РёР·СѓРµРј Рё РїСЂРёРјРµРЅСЏРµРј РЅРµР»РёРЅРµР№РЅРѕСЃС‚Рё
        r = self.bn_r(r)
        p = self.bn_p(p)
        
        # РџСЂРѕРµС†РёСЂСѓРµРј РІ РЅСѓР¶РЅСѓСЋ СЂР°Р·РјРµСЂРЅРѕСЃС‚СЊ
        r = self.projection_r(r)
        p = self.projection_p(p)
        
        # РћР±СЉРµРґРёРЅСЏРµРј РїСЂРёР·РЅР°РєРё С‡РµСЂРµР· СЃР»РѕР¶РµРЅРёРµ РІРјРµСЃС‚Рѕ РєРѕРЅРєР°С‚РµРЅР°С†РёРё
        freq_features = r + p  # [batch, out_channels, modes1]
        
        return freq_features, out_ft


class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()
        self.width = configs.input_channels
        self.channel = configs.input_channels
        self.fl = configs.sequence_len
        
        # РЈР»СѓС‡С€РµРЅРЅС‹Р№ РІС…РѕРґРЅРѕР№ СЃР»РѕР№
        self.fc0 = nn.Sequential(
            nn.Linear(self.channel, configs.mid_channels),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )
        
        # РЈР»СѓС‡С€РµРЅРЅС‹Р№ РїРРІС‹Р№ Р±Р»РѕРє СЃРІРµСЂС‚РєРё
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.Conv1d(configs.mid_channels, configs.mid_channels, kernel_size=3,
                      stride=1, bias=False, padding=1),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        # РЈР»СѓС‡С€РµРЅРЅС‹Р№ РІС‚РѕСЂРѕР№ Р±Р»РѕРє СЃРІРµСЂС‚РєРё
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels, kernel_size=8, 
                      stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.Conv1d(configs.mid_channels, configs.mid_channels, kernel_size=5,
                      stride=1, bias=False, padding=2),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        # РЈР»СѓС‡С€РµРЅРЅС‹Р№ С‚СЂРµС‚РёР№ Р±Р»РѕРє СЃРІРµСЂС‚РєРё
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.final_out_channels, kernel_size=8, 
                      stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.Conv1d(configs.final_out_channels, configs.final_out_channels, kernel_size=5,
                      stride=1, bias=False, padding=2),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )
        
        # РђРґР°РїС‚РёРІРЅС‹Р№ РїСѓР»РёРЅРі СЃ Р±РѕР»СЊС€РёРј РІС‹С…РѕРґРЅС‹Рј СЂР°Р·РјРµСЂРѕРј
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len * 2)
        
        # Р”РѕРїРѕР»РЅРёС‚РµР»СЊРЅС‹Р№ СЃР»РѕР№ РґР»СЏ СѓРумРµРЅСЊС€РµРЅРёСЏ СЂР°Р·РјРµСЂРЅРѕСЃС‚Рё
        self.fc1 = nn.Sequential(
            nn.Linear(configs.final_out_channels * configs.features_len * 2, 
                     configs.final_out_channels * configs.features_len),
            nn.BatchNorm1d(configs.final_out_channels * configs.features_len),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )
        
    def forward(self, x):
        # РџСЂРёРјРµРЅСЏРµРј РІС…РѕРґРЅРѕР№ СЃР»РѕР№
        x = self.fc0(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # РџСЂРёРјРµРЅСЏРµРј Р±Р»РѕРєРё СЃРІРµСЂС‚РєРё
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # РџСЂРёРјРµРЅСЏРµРј Р°РґР°РїС‚РёРІРЅС‹Р№ РїСѓР»РёРЅРі
        x = self.adaptive_pool(x)
        
        # РџСЂРµРѕР±СЂР°Р·СѓРµРј РІ РїР»РѕСЃРєРёР№ РІРµРєС‚РѕСЂ
        x_flat = x.reshape(x.shape[0], -1)
        
        # РџСЂРёРјРµРЅСЏРµРј РґРѕРїРѕР»РЅРёС‚РµР»СЊРЅС‹Р№ СЃР»РѕР№
        x_flat = self.fc1(x_flat)
        
        return x_flat

class tf_encoder(nn.Module):
    def __init__(self, configs):
        super(tf_encoder, self).__init__()
        
        # РЅРёС†РёР°Р»РёР·РёСЂСѓРµРј РІРµСЃР° РґР»СЏ С‡Р°СЃС‚РѕС‚РЅРѕРіРѕ РїСЂРµРѕР±СЂР°Р·РѕРІР°РЅРёСЏ
        self.weights1 = nn.Parameter(torch.randn(configs.fourier_modes, configs.input_channels))
        self.weights2 = nn.Parameter(torch.randn(configs.fourier_modes, configs.input_channels))
        
        # Batch normalization РґР»СЏ СЂРµР°Р»СЊРЅРѕР№ Рё РјРЅРёРјРѕР№ С‡Р°СЃС‚РµР№
        self.bn_r = nn.BatchNorm1d(configs.fourier_modes)
        self.bn_p = nn.BatchNorm1d(configs.fourier_modes)
        
        # РџСЂРѕРµРєС†РёРѕРЅРЅС‹Рµ СЃР»РѕРё РґР»СЏ СЂРµР°Р»СЊРЅРѕР№ Рё РјРЅРёРјРѕР№ С‡Р°СЃС‚РµР№
        self.projection_r = nn.Sequential(
            nn.Conv1d(configs.fourier_modes, configs.mid_channels, kernel_size=1),
            nn.BatchNorm1d(configs.mid_channels)
        )
        self.projection_p = nn.Sequential(
            nn.Conv1d(configs.fourier_modes, configs.mid_channels, kernel_size=1),
            nn.BatchNorm1d(configs.mid_channels)
        )
        
        # РџСЂРѕРµРєС†РёСЏ С‡Р°СЃС‚РѕС‚РЅС‹С… РїСЂРёР·РЅР°РєРѕРІ
        self.freq_projection = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.mid_channels, kernel_size=1),
            nn.BatchNorm1d(configs.mid_channels)
        )
        
        # Р'СЂРµРјРµРЅРЅРѕР№ СЌРЅРєРѕРґРµСЂ
        self.time_encoder = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.Conv1d(configs.mid_channels, configs.mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(configs.mid_channels)
        )
        
    def forward(self, x):
        # РџСЂРёРјРµРЅСЏРµРј РїСЂРµРѕР±СЂР°Р·РѕРІР°РЅРёРµ Р¤СѓСЂСЊРµ
        x_ft = torch.fft.fft(x, dim=2)
        
        # РЅРёР·РІР»РµРєР°РµРј СЂРµР°Р»СЊРЅСѓСЋ Рё РјРЅРёРјСѓСЋ С‡Р°СЃС‚Рё
        real = x_ft.real
        imag = x_ft.imag
        
        # РџСЂРёРјРµРЅСЏРµРј РІРµСЃР° Рё batch normalization
        real = self.bn_r(torch.matmul(self.weights1, real))
        imag = self.bn_p(torch.matmul(self.weights2, imag))
        
        # РџСЂРѕРµС†РёСЂСѓРµРј СЂРµР°Р»СЊРЅСѓСЋ Рё РјРЅРёРјСѓСЋ С‡Р°СЃС‚Рё
        real = self.projection_r(real)
        imag = self.projection_p(imag)
        
        # РћР±СЉРµРґРёРЅСЏРµРј С‡Р°СЃС‚РѕС‚РЅС‹Рµ РїСЂРёР·РЅР°РєРё
        freq_features = torch.cat([real, imag], dim=1)
        freq_features = self.freq_projection(freq_features)
        
        # РЅРёР·РІР»РµРєР°РµРј РІСЂРµРјРµРЅРЅС‹Рµ РїСЂРёР·РЅР°РєРё
        time_features = self.time_encoder(x)
        
        # РћР±СЉРµРґРёРЅСЏРµРј РїСЂРёР·РЅР°РєРё
        features = torch.cat([freq_features, time_features], dim=1)
        features = features.mean(dim=2)  # [batch, mid_channels * 2]
        
        return features, x_ft

class tf_decoder(nn.Module):
    def __init__(self, configs):
        super(tf_decoder, self).__init__()
        
        # РџСЂРѕРµРєС†РёРѕРЅРЅС‹Р№ СЃР»РѕР№
        self.projection = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.mid_channels, kernel_size=1),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU()
        )
        
        # РЎР»РѕР№ СЂРµРєРѕРЅСЃС‚СЂСѓРєС†РёРё
        self.reconstruction = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.input_channels, kernel_size=1),
            nn.BatchNorm1d(configs.input_channels)
        )
        
    def forward(self, features):
        # РџСЂРѕРµС†РёСЂСѓРµРј РїСЂРёР·РЅР°РєРё
        x = self.projection(features)
        
        # РµРєРѕРЅСЃС‚СЂСѓРёСЂСѓРµРј РІСЂРµРјРµРЅРЅРѕР№ СЂСЏРґ
        x = self.reconstruction(x)
        
        return x

class SinkhornDistance(nn.Module):
    def __init__(self, eps=1e-4, max_iter=2000, reduction='sum'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # Compute cost matrix
        C = torch.cdist(x, y, p=2)
        
        # Sinkhorn iterations
        n, m = x.size(0), y.size(0)
        a = torch.ones(n, device=x.device) / n
        b = torch.ones(m, device=x.device) / m
        
        # Normalize cost matrix
        C = C / torch.mean(C)
        
        # Sinkhorn iterations
        K = torch.exp(-C / self.eps)
        
        # Initialize dual variables
        u = torch.zeros_like(a)
        v = torch.zeros_like(b)
        
        # Sinkhorn iterations
        for _ in range(self.max_iter):
            u = a / (K @ v + 1e-8)
            v = b / (K.t() @ u + 1e-8)
        
        # Transport plan
        P = torch.diag(u) @ K @ torch.diag(v)
        
        # Transport cost
        cost = torch.sum(P * C)
        
        if self.reduction == 'mean':
            cost = cost / (n * m)
            
        return cost, u, v

class RAINCOAT(Algorithm):
    def __init__(self, configs):
        super(RAINCOAT, self).__init__(configs)
        
        # РЅРёС†РёР°Р»РёР·РёСЂСѓРµРј feature extractor
        self.feature_extractor = tf_encoder(configs)
        
        # РЅРёС†РёР°Р»РёР·РёСЂСѓРµРј decoder
        self.decoder = tf_decoder(configs)
        
        # РЅРёС†РёР°Р»РёР·РёСЂСѓРµРј РєР»Р°СЃСЃРёС„РёРєР°С‚РѕСЂ
        self.classifier = nn.Sequential(
            nn.Linear(configs.mid_channels * 2, configs.num_classes)
        )
        
        # РЅРёС†РёР°Р»РёР·РёСЂСѓРµРј РѕРїС‚РёРјРёР·Р°С‚РѕСЂ
        self.optimizer = torch.optim.Adam(self.parameters(), lr=configs.lr)
        
        # РЅРёС†РёР°Р»РёР·РёСЂСѓРµРј С„СѓРЅРєС†РёРё РїРѕС‚РµСЂСЊ
        self.criterion = nn.CrossEntropyLoss()
        self.reconstruction_criterion = nn.MSELoss()
        
        # РЅРёС†РёР°Р»РёР·РёСЂСѓРµРј Sinkhorn distance
        self.sinkhorn = SinkhornDistance(eps=0.01, max_iter=100, reduction='mean')
        
    def encode_features(self, x):
        # Извлекаем признаки
        features, _ = self.feature_extractor(x)
        return features  # Возвращаем только features
        
    def update(self, src_x, src_y, trg_x):
        # Извлекаем признаки
        src_features = self.encode_features(src_x)
        trg_features = self.encode_features(trg_x)
        
        # Реконструируем исходные данные
        src_recon = self.decoder(src_features.unsqueeze(-1))
        trg_recon = self.decoder(trg_features.unsqueeze(-1))
        
        # Р'С‡РёСЃР»СЏРµРј РїРѕС‚РµСЂРё СЂРµРєРѕРЅСЃС‚СЂСѓРєС†РёРё
        recon_loss = self.reconstruction_criterion(src_recon, src_x) + \
                    self.reconstruction_criterion(trg_recon, trg_x)
        
        # Р'С‡РёСЃР»СЏРµРј Sinkhorn loss
        sinkhorn_loss, _, _ = self.sinkhorn(src_features, trg_features)
        
        # РљР»Р°СЃСЃРёС„РёС†РёСЂСѓРµРј РёСЃС…РѕРґРЅС‹Рµ РґР°РЅРЅС‹Рµ
        src_pred = self.classifier(src_features)
        
        # Р'С‡РёСЃР»СЏРµРј РїРѕС‚РµСЂРё РєР»Р°СЃСЃРёС„РёРєР°С†РёРё
        cls_loss = self.criterion(src_pred, src_y)
        
        # Р'С‡РёСЃР»СЏРµРј РєРѕРЅС‚СЂР°СЃС‚РЅСѓСЋ РїРѕС‚РµСЂСЋ
        contrastive_loss = self.contrastive_loss(src_features, trg_features)
        
        # РћР±С‰Р°СЏ РїРѕС‚РµСЂСЏ
        loss = recon_loss + sinkhorn_loss + cls_loss + contrastive_loss
        
        # РћРїС‚РёРјРёР·Р°С†РёСЏ
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'recon_loss': recon_loss.item(),
            'sinkhorn_loss': sinkhorn_loss.item(),
            'cls_loss': cls_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'total_loss': loss.item()
        }
        
    def contrastive_loss(self, src_features, trg_features):
        # РќРѕСЂРјР°Р»РёР·СѓРµРј РїСЂРёР·РЅР°РєРё
        src_features = F.normalize(src_features, dim=1)
        trg_features = F.normalize(trg_features, dim=1)
        
        # Р'С‡РёСЃР»СЏРµРј РјР°С‚СЂРёС†Сѓ СЃС…РѕРґСЃС‚РІР°
        similarity = torch.matmul(src_features, trg_features.t())
        
        # РЎРѕР·РґР°РµРј РїРѕР»РѕР¶РёС‚РµР»СЊРЅС‹Рµ РїР°СЂС‹ (РґРёР°РіРѕРЅР°Р»СЊ)
        positive_pairs = torch.diag(similarity)
        
        # РЎРѕР·РґР°РµРј РѕС‚СЂРёС†Р°С‚РµР»СЊРЅС‹Рµ РїР°СЂС‹ (РІСЃРµ РѕСЃС‚Р°Р»СЊРЅС‹Рµ)
        negative_pairs = similarity - torch.eye(similarity.size(0)).to(similarity.device) * positive_pairs.unsqueeze(1)
        
        # Р'С‡РёСЃР»СЏРµРј InfoNCE loss
        loss = -torch.mean(positive_pairs - torch.logsumexp(negative_pairs, dim=1))
        
        return loss
        
    def correct(self, src_x, src_y, trg_x):
        """
        Метод для фазы коррекции, который обновляет веса модели для улучшения классификации.
        """
        # Извлекаем признаки
        src_features = self.encode_features(src_x)
        trg_features = self.encode_features(trg_x)
        
        # Классифицируем исходные данные
        src_pred = self.classifier(src_features)
        
        # Р'С‡РёСЃР»СЏРµРј РїРѕС‚РµСЂРё РєР»Р°СЃСЃРёС„РёРєР°С†РёРё
        cls_loss = self.criterion(src_pred, src_y)
        
        # Р'С‡РёСЃР»СЏРµРј РєРѕРЅС‚СЂР°СЃС‚РЅСѓСЋ РїРѕС‚РµСЂСЋ
        contrastive_loss = self.contrastive_loss(src_features, trg_features)
        
        # РћР±С‰Р°СЏ РїРѕС‚РµСЂСЏ
        loss = cls_loss + contrastive_loss
        
        # РћРїС‚РёРјРёР·Р°С†РёСЏ
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'cls_loss': cls_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'total_loss': loss.item()
        }
