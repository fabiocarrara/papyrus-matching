import argparse

import torch

from train import LitPapyrusAE


def main(args):
    print('Loading checkpoint:', args.ckpt)
    model = LitPapyrusAE.load_from_checkpoint(checkpoint_path=args.ckpt)
    encoder = model.encoder.eval().cpu()
    print('Exporting encoder:', args.output)
    torch.save(encoder, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export Matcher to .pth')
    parser.add_argument('ckpt', help='path to checkpoint to export')
    parser.add_argument('-o', '--output', type=str, default='patch-encoder.pth', help='Output .pth file')
    args = parser.parse_args()
    main(args)