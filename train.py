import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
import torchmetrics.functional as tm

from dataloader import PapyrMatchesDataset


class PapyriMatchesDataModule(pl.LightningDataModule):
    def __init__(self, root='data/fragments/', batch_size=8):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),  # RGBA
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


class LitPapyrusAE(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()

        # Vanilla ConvNet Encoder
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

        # Vanilla ConvNet Decoder
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
        return self.encoder(x).flatten(start_dim=1)
    
    def _common_step(self, stage, batch, batch_idx):
        left, right = batch
        n_samples = left.shape[0]
        device = left.device

        # Vanilla Autoencoder
        enc_l = self.encoder(left)
        enc_r = self.encoder(right)

        dec_l = self.decoder(enc_l)
        dec_r = self.decoder(enc_r)

        # Reconstruction (L2) Loss Term
        reconstruct_loss = 0.5 * torch.mean((left - dec_l)**2) + 0.5 * torch.mean((right - dec_r)**2)

        left_codes  = F.normalize(enc_l.flatten(start_dim=1))
        right_codes = F.normalize(enc_r.flatten(start_dim=1))
        logits = torch.matmul(left_codes, right_codes.T) / self.temperature
        y_true = torch.arange(n_samples, device=device)

        # Contrastive Loss Term
        contrastive_loss = F.cross_entropy(logits, y_true)

        # Loss
        loss = contrastive_loss + 100 * reconstruct_loss
        
        # Metrics
        accuracy = tm.accuracy(logits, y_true)

        y_scores = logits.ravel()
        y_true = torch.eye(n_samples, dtype=torch.long, device=device).ravel()
        auroc = tm.auroc(y_scores, y_true)

        # Log stuff
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

    def training_step(self, batch, batch_idx):
        return self._common_step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._common_step('val', batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self._common_step('test', batch, batch_idx)
    
    def _common_epoch_end(self, stage, step_outputs):
        keys = list(step_outputs[0].keys())
        metrics = {key: [i[key].detach().cpu() for i in step_outputs] for key in keys}

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
        self._common_epoch_end('train', train_step_outputs)

    def validation_epoch_end(self, validation_step_outputs):
        self._common_epoch_end('val', validation_step_outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-7)

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
        resume_from_checkpoint=resume,
        default_root_dir=run_dir,
        max_epochs=args['epochs'],
        gpus=1,
        deterministic=True,
        callbacks=[
            ModelCheckpoint(monitor="val/auroc", save_last=True),
            LearningRateMonitor(logging_interval='step'),
        ]
    )

    trainer.fit(scorer, dm)
    trainer.test(scorer, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Papyrus Match Scorer')
    parser.add_argument('-e', '--epochs', type=int, default=15, help='number of training epochs')
    parser.add_argument('-r', '--resume', default=False, action='store_true', help='resume training')

    args = parser.parse_args()
    args = vars(args)
    main(args)