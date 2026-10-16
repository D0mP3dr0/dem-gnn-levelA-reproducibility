import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.utils.metrics import compute_p90_metrics, compute_physics_consistency, compute_morphology_metrics, compute_texture_metrics
from src.config import STATS


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act2 = nn.GELU()
        
        self.shortcut = nn.Identity()
        if in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, 1)
            
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out += residual
        return out

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class DeepResAttUNet(nn.Module):
    def __init__(self, in_channels=17, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.deep_supervision = True
        
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        curr_in = in_channels
        for feature in features:
            self.downs.append(ResidualBlock(curr_in, feature))
            curr_in = feature
            
        self.bottleneck = ResidualBlock(features[-1], features[-1]*2)
        
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        
        curr_g = features[-1]*2
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(curr_g, feature, 2, 2))
            
            self.attentions.append(AttentionBlock(F_g=feature, F_l=feature, F_int=feature//2))
            
            self.ups.append(ResidualBlock(feature*2, feature))
            
            curr_g = feature
            
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)        
        
        if self.deep_supervision:
            
            self.ds_head_1 = nn.Conv2d(features[2], out_channels, 1)
            
            self.ds_head_2 = nn.Conv2d(features[1], out_channels, 1)
            
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        decoder_outputs = []
        
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            
            skip = skip_connections[idx//2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            
            skip = self.attentions[idx//2](g=x, x=skip)
            
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](concat_skip)
            
            decoder_outputs.append(x)
            
        final = self.final_conv(x)
        
        if self.deep_supervision and self.training:
            
            d1 = self.ds_head_1(decoder_outputs[1])
            d1 = F.interpolate(d1, size=final.shape[2:], mode='bilinear', align_corners=False)
            
            d2 = self.ds_head_2(decoder_outputs[2])
            d2 = F.interpolate(d2, size=final.shape[2:], mode='bilinear', align_corners=False)
            
            return final, d2, d1
            
        return final


class GeometricConsistencyLoss(nn.Module):
    def __init__(self, tv_weight=1e-4):
        super().__init__()
        self.tv_weight = tv_weight

    def forward(self, pred, input_dem, mask):
        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        
        dy_dem = input_dem[:, :, 1:, :] - input_dem[:, :, :-1, :]
        dx_dem = input_dem[:, :, :, 1:] - input_dem[:, :, :, :-1]
        
        gap_mask = 1.0 - mask
        gap_mask_y = gap_mask[:, :, 1:, :]
        gap_mask_x = gap_mask[:, :, :, 1:]
        
        loss_grad = (F.huber_loss(dy_pred, dy_dem, reduction='none') * gap_mask_y).mean() + \
                    (F.huber_loss(dx_pred, dx_dem, reduction='none') * gap_mask_x).mean()
        
        loss_tv = torch.sum(torch.abs(dy_pred)) + torch.sum(torch.abs(dx_pred))
        
        return loss_grad + self.tv_weight * loss_tv


class CNNTrainer(pl.LightningModule):
    def __init__(self, biome='amazonia', lr=1e-4, use_lidar_input=False):
        super().__init__()
        self.save_hyperparameters()
        self.biome = biome
        self.lr = lr
        self.use_lidar_input = use_lidar_input
        
        in_ch = 19 if use_lidar_input else 17
        self.model = DeepResAttUNet(in_channels=in_ch, out_channels=1)
        
        self.geo_loss = GeometricConsistencyLoss()
        self.physics_weight = 0.1 if use_lidar_input else 0.0 
        
        if biome in STATS:
            self.stats = STATS[biome]
        else:
            self.stats = {'mean': 0.0, 'std': 1.0}
            print(f"Warning: Biome {biome} not found in STATS. Using raw values.")
        
    def forward(self, x):
        return self.model(x)
        
    def deep_supervision_loss(self, preds, target, mask, input_dem):
        if isinstance(preds, tuple):
            final, d2, d1 = preds
            w0, w1, w2 = 1.0, 0.5, 0.25
            
            l_lidar_0 = (F.huber_loss(final, target, reduction='none') * mask).sum() / (mask.sum() + 1e-6)
            l_lidar_1 = (F.huber_loss(d2, target, reduction='none') * mask).sum() / (mask.sum() + 1e-6)
            l_lidar_2 = (F.huber_loss(d1, target, reduction='none') * mask).sum() / (mask.sum() + 1e-6)
            
            total_lidar = w0*l_lidar_0 + w1*l_lidar_1 + w2*l_lidar_2
            
            l_physics = self.geo_loss(final, target, mask)
            
            return total_lidar, final
        else:
            loss_lidar = (F.huber_loss(preds, target, reduction='none') * mask).sum() / (mask.sum() + 1e-6)
            return loss_lidar, preds

    def training_step(self, batch, batch_idx):
        x, y = batch 
        lidar_target = y[:, 0:1]
        mask = y[:, 1:2]
        input_dem = x[:, 0:1]
        
        outputs = self.model(x)
        
        target_res = lidar_target - input_dem
        
        if isinstance(outputs, tuple):
            pred_res_final = outputs[0]
            loss_lidar = (F.huber_loss(pred_res_final, target_res, reduction='none') * mask).sum() / (mask.sum() + 1e-6)
        else:
            pred_res_final = outputs
            loss_lidar = (F.huber_loss(pred_res_final, target_res, reduction='none') * mask).sum() / (mask.sum() + 1e-6)
        
        pred_final = input_dem + pred_res_final
        loss_physics = self.geo_loss(pred_final, input_dem, mask)
        
        total_loss = loss_lidar + self.physics_weight * loss_physics
        
        with torch.no_grad():
            mae = (torch.abs(pred_final - lidar_target) * mask).sum() / (mask.sum() + 1e-6)
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_lidar_loss', loss_lidar, on_epoch=True)
        self.log('train_physics_loss', loss_physics, on_epoch=True)
        self.log('train_mae', mae, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        lidar_target = y[:, 0:1]
        mask = y[:, 1:2]
        input_dem = x[:, 0:1]
        
        pred_res = self.model(x)
        pred_final = input_dem + pred_res
        
        loss = (F.huber_loss(pred_res, lidar_target - input_dem, reduction='none') * mask).sum() / (mask.sum() + 1e-6)
        mae = (torch.abs(pred_final - lidar_target) * mask).sum() / (mask.sum() + 1e-6)
        
        std = self.stats['std']
        mae_meters = mae * std
        
        metrics_mae_near_val = mae_meters
        metrics_mae_far_val = mae_meters
        
        if self.use_lidar_input:
            try:
                x_sp = x.clone()
                valid_mask = (mask > 0.5).squeeze(1)
                
                if valid_mask.sum() > 0:
                    noise = torch.rand_like(valid_mask.float())
                    context_mask = valid_mask & (noise < 0.5)
                    
                    x_sp[:, 17, :, :] *= context_mask.float()
                    x_sp[:, 18, :, :] *= context_mask.float()
                    
                    pred_res_sp = self.model(x_sp)
                    pred_final_sp = input_dem + pred_res_sp
                    
                    ctx_4d = context_mask.float().unsqueeze(1)
                    dilated_ctx = F.max_pool2d(ctx_4d, kernel_size=11, stride=1, padding=5)
                    
                    holdout_mask = valid_mask & (~context_mask)
                    near_4d = ((dilated_ctx > 0.5).squeeze(1) & holdout_mask).unsqueeze(1).float()
                    far_4d = ((dilated_ctx < 0.5).squeeze(1) & holdout_mask).unsqueeze(1).float()
                    
                    if near_4d.sum() > 0:
                        metrics_mae_near_val = ((torch.abs(pred_final_sp - lidar_target) * near_4d).sum() / (near_4d.sum() + 1e-6)) * std
                    if far_4d.sum() > 0:
                        metrics_mae_far_val = ((torch.abs(pred_final_sp - lidar_target) * far_4d).sum() / (far_4d.sum() + 1e-6)) * std
            except: pass
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', mae, prog_bar=False) 
        self.log('val_mae_m', mae_meters, prog_bar=True) 
        self.log('mae_near_m', metrics_mae_near_val, prog_bar=False)
        self.log('mae_far_m', metrics_mae_far_val, prog_bar=False)
        
        if metrics_mae_near_val > 1e-6:
            self.log('miopia_ratio', metrics_mae_far_val / metrics_mae_near_val, prog_bar=False)
            
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
