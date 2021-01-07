import socket
import os

from datetime import datetime


class Workspace():
    def __init__(self, args):
        # initial default configuration
        # -----------------------------
        self.config = {
            # global config
            'log_dir': "logs",
            'root_dir': "/path/to/fer/dataset",
            'deterministic': False,
            'seed': 0,
            'workers': 4,
            # training config
            'arch': 'resnet18',
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'batch_size': 64,
            'epochs': 60,
            'pretrained': "msceleb",
            # centerloss config
            'alpha': 0.5,
            'lamb': 0.01
        }

        # over-ride if CL args are given
        self.override(args)

        # tag for filename and comet.ml experiment
        # ----------------------------------------
        self.tag = (
            f'_a-centerloss'
            f'_ar={self.config["arch"]}'
            f'_pt={self.config["pretrained"]}'
            f'_bs={self.config["batch_size"]}'
            f'_lr={self.config["lr"]}'
            f'_wd={self.config["weight_decay"]}'
            f'_alpha={self.config["alpha"]}'
            f'_lamb={self.config["lamb"]}'
        )

        # setup writers
        self.setup()

    def override(self, args):
        '''
        over-ride config if arguments are given
        '''
        if args.arch is not None:
            self.config['arch'] = args.arch
        if args.bs is not None:
            self.config['batch_size'] = args.bs
        if args.lr is not None:
            self.config['lr'] = args.lr
        if args.wd is not None:
            self.config['weight_decay'] = args.wd
        if args.epochs is not None:
            self.config['epochs'] = args.epochs
        if args.alpha is not None:
            self.config['alpha'] = args.alpha
        if args.lamb is not None:
            self.config['lamb'] = args.lamb
        if args.pretrained is not None:
            self.config['pretrained'] = args.pretrained
        self.config['deterministic'] = args.deterministic

    def setup(self):
        '''
        setup log names and save locations
        '''
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        logname = current_time + '_' + socket.gethostname() + self.tag
        self.config['save_path'] = os.path.join(self.config['log_dir'], logname)
