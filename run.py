import argparse
import os
import warnings
import torch
import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric
import matplotlib.pyplot as plt
import tensorboardX
import wandb

def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    phase_logger.info('Create the dataloader for {}.'.format(opt['phase']))
    phase_logger.info('The batch size is {}.'.format(opt['datasets'][opt['phase']]['dataloader']['args']['batch_size']))
    phase_logger.info('len(phase_loader): {}'.format(len(phase_loader)))
    phase_logger.info('len(val_loader): {}'.format(len(val_loader) if val_loader is not None else 0))
    
    # validate dataset
    if opt['validate_data']:
        print("validate data")
        for i, data in enumerate(phase_loader):
            image_gt = data['gt_image'][0]
            noisy_image = data['cond_image'][0]
            # print(image_gt.shape, noisy_image.shape)
            np_gt = image_gt.permute(1,2,0).numpy()*0.5 + 0.5
            np_noise = noisy_image.permute(1,2,0).numpy()*0.5 + 0.5
            plt.subplot(1,2,1)
            plt.imshow(np_gt)
            plt.subplot(1,2,2)
            plt.imshow(np_noise)
            # save image
            plt.savefig("/home/awi-docker/image_quality/Palette-Image-to-Image-Diffusion-Models/experiments/validate_data_{}.png".format(i))
            if i > 10:
                break
    
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    # ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]
    
    # print("done loading network, metrics, losses")

    model = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )
    phase_logger.info('Begin model {}.'.format(opt['phase']))
    
    try:
        if opt['phase'] == 'train':
            model.train()
        else:
            model.test()
    finally:
        phase_writer.close()
    
        
if __name__ == '__main__':
    wandb.init(project="diffusion_64*64_image_quality_enhancement", sync_tensorboard=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/image_quality.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='test')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('-v', '--validate_data', action='store_true')

    ''' parser configs '''
    args = parser.parse_args()
    opt = Praser.parse(args)
    
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1 
        main_worker(0, 1, opt)
        
    wandb.finish()