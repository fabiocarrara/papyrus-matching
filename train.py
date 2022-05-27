import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet
from torchvision.transforms import functional as TF
import torchmetrics.functional as tm

from dataloader import PapyrMatchesDataset
from models import VisionTransformer, ConvMixer

class RGBA2GrayTransform:
    """ Transform RGBA to GA. """

    def __call__(self, x):
        rgb, alpha = x[..., :3, :, :], x[..., 3:, :, :]
        gray = TF.rgb_to_grayscale(rgb, 1)
        return torch.cat([gray, alpha], axis=-3)


class PapyriMatchesDataModule(pl.LightningDataModule):
    def __init__(self, root='data/fragments/', batch_size=8):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # RGBA
            transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
            # Gray + Alpha
            # RGBA2GrayTransform(),
            # transforms.Normalize([0.5, 0.5], [0.5, 0.5])
        ])

    def setup(self, stage=None):
        # find papyri files
        image_paths = sorted(self.root.rglob('*.png'))

        # split image list
        n_images = len(image_paths)
        n_train = round(n_images * 0.50)
        n_valid = round(n_images * 0.25)
        n_test  = n_images - n_train - n_valid
        
        train_image_paths = image_paths[:n_train]
        valid_image_paths = image_paths[n_train:n_train + n_valid]
        test_image_paths  = image_paths[-n_test:]

        common = dict(
            patch_size=64,
            stride=64,
            rows=1,
            cols=1,
            transform=self.transform,
        )

        self.train_dataset = ConcatDataset([PapyrMatchesDataset(i, **common) for i in train_image_paths])
        self.valid_dataset = ConcatDataset([PapyrMatchesDataset(i, **common) for i in valid_image_paths])
        self.test_dataset  = ConcatDataset([PapyrMatchesDataset(i, **common) for i in test_image_paths ])

        print("train_dataset len:", len(self.train_dataset))
        print("valid_dataset len:", len(self.valid_dataset))
        print( "test_dataset len:", len(self.test_dataset ))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def teardown(self, stage=None):
        # Used to clean-up when the run is finished
        pass


class LitPapyrusScorer(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        ## ViT
        # self.model = VisionTransformer(image_size=(128, 128), in_channels=4, num_layers=7, num_classes=1)
        
        ## ResNet with RGBA input
        self.model = resnet.resnet18(num_classes=1)
        self.model.conv1 = torch.nn.Conv2d(8, 64, kernel_size=3, stride=2, padding=3, bias=False)

        ## ConvMixer
        # self.model = ConvMixer(512, 3, in_channels=4, patch_size=8, num_classes=1)

        ## ResNet RGBA, Dual Encoder
        # self.model = resnet.resnet18(num_classes=1)
        # self.model.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.model.avgpool = torch.nn.Identity()  # remove avgpool layer
        # self.model.fc = torch.nn.Identity()  # remove last fc layer
        # self.temperature = 0.5
        
        self.lr = lr

    def forward(self, x):
        return self.model(x)
    
    def _trainval_step_input_fusion(self, stage, batch, batch_idx):
        left, right = batch
        n_samples = left.shape[0]
        device = left.device

        matches = torch.cat([torch.cat([left, right], axis=1),  # concat images channels
                             torch.cat([right, left], axis=1)], axis=0)
        matches_logits = self.model(matches).squeeze(dim=1)

        if stage == 'val':
            """ A RANDOM NEGATIVE """
            rolled_right = torch.roll(right, 1, 0)
            non_matches = torch.cat([torch.cat([left, rolled_right], axis=1),  # concat images channels
                                    torch.cat([rolled_right, left], axis=1)], axis=0)
            non_matches_logits = self.model(non_matches).squeeze(dim=1)

            """ ALL NEGATIVES
                non_matches_logits = []
                for i in range(1, n_samples):
                    rolled_right = torch.roll(right, i, 0)
                    non_matches = torch.cat([torch.cat([left, rolled_right], axis=1),
                                            torch.cat([rolled_right, left], axis=1)], axis=0)
                    non_matches_logits.append( self.model(non_matches).squeeze(dim=1) )
                non_matches_logits = torch.cat(non_matches_logits)
            """

        else:
            """ HARD/EASY NEGATIVES """
            with torch.no_grad():
                dmat_lr = torch.full((n_samples, n_samples), float('+inf'), dtype=torch.float32, device=device)
                dmat_rl = torch.full((n_samples, n_samples), float('+inf'), dtype=torch.float32, device=device)

                # # batched-version
                # rows = torch.arange(n_samples, device=device)
                # for i in range(1, n_samples):
                #    cols = (rows + i) % n_samples
                #    x = torch.cat([left, torch.roll(right, i, 0)], axis=-1)
                #    dmat_lr[rows, cols] = self.model(x).squeeze(dim=1)
                #
                #    x = torch.cat([torch.roll(right, i, 0), left], axis=-1)
                #    dmat_rl[rows, cols] = self.model(x).squeeze(dim=1)

                # non-batched version
                for i, l_i in enumerate(left):
                    for j, r_j in enumerate(right):
                        if i == j:
                            continue
                        x = torch.cat([l_i, r_j], axis=0).unsqueeze(dim=0)
                        dmat_lr[i, j] = self.model(x).squeeze()
                        x = torch.cat([r_j, l_i], axis=0).unsqueeze(dim=0)
                        dmat_rl[j, i] = self.model(x).squeeze()

                hard_right = torch.argmin(dmat_lr, 1)
                hard_left = torch.argmin(dmat_rl, 1)
            
            non_matches = torch.cat([torch.cat([left, right[hard_right]], axis=1),  # concat images channels
                                    torch.cat([right, left[hard_left]], axis=1)], axis=0)
            non_matches_logits = self.model(non_matches).squeeze(dim=1)

        if True and batch_idx == 0:
            # pos_images = matches[:, :-1, ...].expand(-1, 3, -1, -1) * 0.5 + 0.5
            # neg_images = non_matches[:, :-1, ...].expand(-1, 3, -1, -1) * 0.5 + 0.5

            pos_images = torch.cat([matches[:, :3, ...], matches[:, 4:-1, ...]], axis=-1).expand(-1, 3, -1, -1) * 0.5 + 0.5
            neg_images = torch.cat([non_matches[:, :3, ...], non_matches[:, 4:-1, ...]], axis=-1).expand(-1, 3, -1, -1) * 0.5 + 0.5

            self.logger.experiment.add_images(f'{stage}/inputs-pos', pos_images, self.current_epoch)
            self.logger.experiment.add_images(f'{stage}/inputs-neg', neg_images, self.current_epoch)

        logits = torch.cat([matches_logits, non_matches_logits])
        y_true = torch.cat([torch.ones_like(matches_logits), torch.zeros_like(non_matches_logits)])

        loss = F.binary_cross_entropy_with_logits(logits, y_true)

        y_scores = torch.sigmoid(logits)
        
        auroc = tm.auroc(logits, y_true.long())
        accuracy = ((y_scores >= 0.5) == y_true).float().mean()

        common = dict(on_epoch=True, prog_bar=True)
        self.log(f'{stage}_loss', loss, **common)
        self.log(f'{stage}_accuracy', accuracy, **common)
        self.log(f'{stage}_auroc', auroc, **common)

        return {
            'loss': loss,
            'y_scores': y_scores,
            'y_true': y_true
        }

    def _trainval_step_dual_encoder(self, stage, batch, batch_idx):
        left, right = batch
        n_samples = left.shape[0]
        device = left.device

        left_features  = F.normalize(self.model(left ))
        right_features = F.normalize(self.model(right))

        logits = torch.matmul(left_features, right_features.T) / self.temperature
        y_true = torch.arange(n_samples, device=device)

        loss = F.cross_entropy(logits, y_true)
        
        accuracy = tm.accuracy(logits, y_true)

        y_scores = logits.ravel()
        y_true = torch.eye(n_samples, dtype=torch.long, device=device).ravel()
        auroc = tm.auroc(y_scores, y_true)

        self.log(f'{stage}_loss', loss, on_epoch=True)
        self.log(f'{stage}_accuracy', accuracy, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_auroc', auroc, on_epoch=True, prog_bar=True)

        return {
            'loss': loss,
            'y_scores': y_scores,
            'y_true': y_true
        }
    
    def _trainval_step(self, *args, **kwargs):
        return self._trainval_step_input_fusion(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self._trainval_step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._trainval_step('val', batch, batch_idx)
    
    def _trainval_epoch_end(self, stage, step_outputs):
        n_max = 10000 // step_outputs[0]['y_scores'].shape[0]
        y_scores = torch.cat([i['y_scores'] for i in step_outputs[:n_max]]).detach().cpu().numpy()
        y_true = torch.cat([i['y_true'] for i in step_outputs[:n_max]]).detach().cpu().numpy()
        figure = sns.histplot(x=y_scores, hue=y_true).get_figure()
        self.logger.experiment.add_figure(f'{stage}/scores', figure, self.current_epoch)

    def training_epoch_end(self, train_step_outputs):
        self._trainval_epoch_end('train', train_step_outputs)

    def validation_epoch_end(self, validation_step_outputs):
        self._trainval_epoch_end('val', validation_step_outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        # optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)

        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'monitor': 'val_loss',
                'frequency': 1
            }
        }


class LitPapyrusAE(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        ## ViT
        # self.model = VisionTransformer(image_size=(128, 128), in_channels=4, num_layers=7, num_classes=1)
        
        ## ResNet with RGBA input
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 16, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
        )
        
        self.lr = lr
        self.temperature = 1

    def forward(self, x):
        return self.encoder(x)
    
    def _trainval_step_ae(self, stage, batch, batch_idx):
        left, right = batch
        n_samples = left.shape[0]
        device = left.device

        enc_l = self.encoder(left)
        enc_r = self.encoder(right)

        dec_l = self.decoder(enc_l)
        dec_r = self.decoder(enc_r)

        reconstruct_loss = 0.5 * torch.mean((left - dec_l)**2) + 0.5 * torch.mean((right - dec_r)**2)

        left_codes  = F.normalize(enc_l.flatten(start_dim=1))
        right_codes = F.normalize(enc_r.flatten(start_dim=1))

        logits = torch.matmul(left_codes, right_codes.T) / self.temperature
        y_true = torch.arange(n_samples, device=device)

        contrastive_loss = F.cross_entropy(logits, y_true)

        loss = contrastive_loss + 100*reconstruct_loss
        
        accuracy = tm.accuracy(logits, y_true)

        y_scores = logits.ravel()
        y_true = torch.eye(n_samples, dtype=torch.long, device=device).ravel()
        auroc = tm.auroc(y_scores, y_true)

        if stage == 'train':
            self.log(f'{stage}/loss', loss)
            self.log(f'{stage}/rloss', reconstruct_loss)
            self.log(f'{stage}/closs', contrastive_loss)
            self.log(f'{stage}/accuracy', accuracy, prog_bar=True)
            self.log(f'{stage}/auroc', auroc, prog_bar=True)

        orig = torch.cat((left[:3,:3], right[:3,:3]), dim=0)
        recon = torch.cat((dec_l[:3,:3], dec_r[:3,:3]), dim=0)
        self.logger.experiment.add_images(f'{stage}/input', orig, self.current_epoch)
        self.logger.experiment.add_images(f'{stage}/recon', recon, self.current_epoch)

        return {
            'loss': loss,
            'rloss': reconstruct_loss,
            'closs': contrastive_loss,
            'y_scores': y_scores,
            'y_true': y_true
        }

    def _trainval_step(self, *args, **kwargs):
        return self._trainval_step_ae(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self._trainval_step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._trainval_step('val', batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self._trainval_step('test', batch, batch_idx)
    
    def _trainval_epoch_end(self, stage, step_outputs):
        keys = list(step_outputs[0].keys())
        # n_max = 10000 // step_outputs[0]['y_scores'].shape[0]
        n_max = None
        metrics = {key: [i[key].detach().cpu() for i in step_outputs[:n_max]] for key in keys}

        y_scores = torch.cat(metrics['y_scores'])
        y_true = torch.cat(metrics['y_true'])

        loss = torch.mean(torch.stack(metrics['loss']))
        reconstruct_loss = torch.mean(torch.stack(metrics['rloss']))
        contrastive_loss = torch.mean(torch.stack(metrics['closs']))
        accuracy = tm.accuracy(y_scores, y_true)
        auroc = tm.auroc(y_scores, y_true)

        self.log(f'{stage}/loss', loss)
        self.log(f'{stage}/rloss', reconstruct_loss)
        self.log(f'{stage}/closs', contrastive_loss)
        self.log(f'{stage}/accuracy', accuracy, prog_bar=True)
        self.log(f'{stage}/auroc', auroc, prog_bar=True)

        figure = sns.histplot(x=y_scores, hue=y_true).get_figure()
        self.logger.experiment.add_figure(f'{stage}/scores', figure, self.current_epoch)

    def training_epoch_end(self, train_step_outputs):
        self._trainval_epoch_end('train', train_step_outputs)

    def validation_epoch_end(self, validation_step_outputs):
        self._trainval_epoch_end('val', validation_step_outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr) #, weight_decay=0.1)
        # optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)

        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'monitor': 'val_loss',
                'frequency': 1
            }
        }


def main(args):
    seed_everything(42, workers=True)

    run_dir = 'runs/'

    dm = PapyriMatchesDataModule(batch_size=8)
    scorer = LitPapyrusAE(lr=1e-3)

    resume = None
    if args.get('resume', False):
        ckpts = Path(run_dir).glob('lightning_logs/version_*/checkpoints/*.ckpt')
        ckpts = sorted(ckpts, reverse=True, key=lambda p: p.stat().st_mtime)
        resume = ckpts[0] if ckpts else None

    trainer = Trainer(
        # track_grad_norm=2,
        # limit_train_batches=250,
        resume_from_checkpoint=resume,
        default_root_dir=run_dir,
        max_epochs=args['epochs'],
        gpus=1,
        deterministic=True,
        callbacks=[
            ModelCheckpoint(monitor="val/auroc", save_last=True),
            LearningRateMonitor(logging_interval='step'),
            # StochasticWeightAveraging()
        ]
    )

    trainer.fit(scorer, dm)
    trainer.test(scorer, datamodule=dm)

    """
    ckpts = Path(run_dir).glob('lightning_logs/version_*/checkpoints/*.ckpt')
    ckpt_dir = sorted(ckpts, reverse=True, key=lambda p: p.stat().st_mtime)[0].parent
    best_ckpt = Path(ckpt_dir).glob('epoch*.ckpt')
    best_ckpt = next(best_ckpt)
    print('Loading:', best_ckpt)
    scorer.load_from_checkpoint(best_ckpt)
    
    test_metrics = trainer.test(scorer, datamodule=dm)
    print(test_metrics)
    breakpoint()
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Papyrus Match Scorer')
    parser.add_argument('-e', '--epochs', type=int, default=15, help='number of training epochs')
    parser.add_argument('-r', '--resume', default=False, action='store_true', help='resume training')

    args = parser.parse_args()
    args = vars(args)
    main(args)