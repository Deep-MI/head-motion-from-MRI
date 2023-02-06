"""
core functions for executing training
"""

import time

import torch
import numpy as np

from modenet.logging import metrics, checkpointing, tensorboard
from modenet import utils

import warnings

def training_loop(args, train_loader, val_loader, model, loss_function, optimizer, writer, 
                  softlabel_helper=None, logdir='tensorboard_logs'):
    """
    executes desired number of epochs with some logging

    for arguments see train.py 
    """


    best_precision = -1e6
    best_spr_rho = 0
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        start_time = time.time()

        _ = run_epoch(train_loader, model, loss_function, epoch, train=True, optimizer=optimizer, 
                summary_writer=writer, patch_mode=False, softlabel_helper=softlabel_helper)

        # evaluate on validation set
        accur_val, spearmanRho = run_epoch(val_loader, model, loss_function, epoch, train=False, 
                summary_writer=writer, patch_mode=args.patching, softlabel_helper=softlabel_helper)

        if np.isnan(accur_val):
            warnings.warn('NaN value encountered in validation accuracy. Stopping training.')
            break

        # remember best precision and save checkpoint
        if accur_val > best_precision:
            print('=> new best accuracy  ', accur_val,' R2 score')
            best_precision = accur_val
            best_spr_rho = spearmanRho
            is_best = True

        else:
            print('previous best accuracy', best_precision, 'current accuracy', accur_val)
            is_best = False

        if logdir is not None:
            checkpointing.saveCheckpoint({
                    'epoch': epoch,
                    'arch': args,
                    'state_dict': model.state_dict(),
                    'best_precision' : best_precision,
                    'optimizer' : optimizer.state_dict(),
                    #'scheduler' : learning_rate.state_dict()
                    },
                is_best=is_best, logdir=logdir, run_id=args.run_id)

        if args.tensorboard:
            writer.add_scalar('seconds_per_epoch', time.time() - start_time, epoch)
            writer.add_scalar('best_accuracy', best_precision, epoch)
            writer.add_scalar('val_best_spearmans_rho', best_spr_rho, epoch)
        print('Best accuracy', best_precision)


    if args.tensorboard:
        writer.add_text('best_accuracy', str(best_precision) + 'with spearmans rho of ' + str(best_spr_rho))
        writer.close()


def run_epoch(data_loader, model, loss_function, epoch, train, patch_mode=False, softlabel_helper=None, optimizer=None, 
              summary_writer=None, scheduler=None, print_freq=1, train_str=None):
    """
    run one epoch of network training, including logging and evaluation on validation set
    """

    # record results for all iterations in these
    batch_time = metrics.AverageMeter()
    losses = metrics.AverageMeter()
    scores = metrics.ScoreKeeper()

    if summary_writer is not None:
        use_tensorboard = True
    else:
        use_tensorboard = False

    # set torch.no_grad for faster execution in case of evaluation
    if train:
        context_manager = utils.misc.EmptyContextManager()
        assert(optimizer is not None)
        model.train()
        if not train_str:
            train_str = 'training  '  # string used to prepend to logs
    else:
        context_manager = torch.torch.no_grad()
        model.eval()
        if not train_str:
            train_str = 'validation'  # string used to prepend to logs

    if patch_mode:
        patch_scorer = metrics.PatchScoreKeeper()

    if softlabel_helper is not None:
        softlabels = True
    else:
        softlabels = False

    end = time.time()

    with context_manager:  # apply torch.no_grad() if necessary

        for i, sample in enumerate(data_loader):

            if use_tensorboard and i == 0 and epoch % 50 == 0:  # do in both train and test
                images = sample['image'].clone().squeeze(1).detach()  # copy images to log them later
            else: 
                images = None

            input = sample['image'].float().cuda()
            target = sample['label'].float().cuda()  #(async=True)

            output = model(input)  # compute output

            ### automatically adjust image shape from network output and target
            output = utils.functional.unsqueeze_to_dimension(output, 2)
            target = utils.functional.unsqueeze_to_dimension(target, 2)

            # expect 2 dimensional labels
            if output.shape != target.shape:
                target = target.T
                if output.shape != target.shape:
                    warnings.warn('output and target shape mismatch')


            if torch.isnan(output).any():
                warnings.warn('nan in network output')
                import ipdb; ipdb.set_trace()  # if this is reached weights probably exploded - adjust LR
            
            loss = loss_function(output, target)

            if softlabels:
                output = softlabel_helper.softLabelToHardLabel(output.detach().cpu())
                target = sample['hard_label'].detach()
            else:
                output = output.detach()

            

            if use_tensorboard and images is not None:  # do in both train and test
                try:
                    tensorboard.saveInputOutput(epoch, images, output, target, summary_writer, train_str)
                except Exception as e:
                    print('images not saved to logs:', e) # ignore logging errors for now
                del images

            # measure accuracy and record loss
            if patch_mode:
                patch_scorer.update(output, sample)
            losses.update(float(loss), input.size(0))
            scores.update(output, target)

            # compute gradient and do SGD step
            if train:
                optimizer.zero_grad() # set_to_none=True
                loss.backward()
                optimizer.step()
                if scheduler != None:
                    scheduler.step()
                elif summary_writer is not None:
                    summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                try:
                    print('{0} epoch[{1}] step[{2}/{3}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                              train_str, epoch, i+1, len(data_loader), batch_time=batch_time,
                              loss=losses))
                except TypeError: # no length defined - can happen with patch generator
                    print('{0} epoch[{1}] step[{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                              train_str, epoch, i+1, batch_time=batch_time,
                              loss=losses))


        if not np.isinf(scores.output).any():
            r2_score = scores.calculateR2()
        else:
            r2_score = np.nan
        spearman = scores.calculateSpearmanR()
        accuracy = scores.calculateMeanDifference()

        # log to TensorBoard
        if use_tensorboard:
            if patch_mode:
                patch_r2_score = patch_scorer.calculateR2()
                patch_spearman = patch_scorer.calculateSpearmanR()
                patch_accuracy = patch_scorer.calculateMeanDifference()

                summary_writer.add_scalar('patch_MAE/%s' % train_str, patch_accuracy, epoch)
                summary_writer.add_scalar('patch_R2_score/%s' % train_str, patch_r2_score, epoch)
                summary_writer.add_scalar('patch_Spearman/%s' % train_str, patch_spearman, epoch)

            summary_writer.add_scalar('Loss/%s' % train_str, losses.val, epoch)
            summary_writer.add_scalar('MAE/%s' % train_str, accuracy, epoch)
            summary_writer.add_scalar('R2_score/%s' % train_str, r2_score, epoch)
            summary_writer.add_scalar('Spearman/%s' % train_str, spearman, epoch)

        # return accuracy for saving the best model
        return r2_score, spearman