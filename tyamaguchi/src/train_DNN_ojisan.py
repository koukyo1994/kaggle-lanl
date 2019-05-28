import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer import serializers
from chainer.training import extensions
from chainer.training import triggers
from chainer.datasets import split_dataset_random,get_cross_validation_datasets
import numpy as np
import pandas as pd
from pathlib import Path
import os
import shutil
from importlib import import_module
import time
import argparse
import sys
from sklearn.preprocessing import StandardScaler

sys.path.append('.')

DataPath = Path('tyamaguchi/data/ojisan_features_36')
delimiter = ','
out = Path('tyamaguchi/nn_results')
AugFiles=['train_x_aug_0.4.csv']
top_features = pd.read_csv('tyamaguchi/data/ojisan_features/columns_LB1346.csv')['feature'].values

def make_dataset(train_file='train_x.csv', features=None, aug_files=[]):
    df = pd.read_csv(DataPath/train_file)
    df = df.drop(['seg_id','seg_start','seg_end'],axis=1)
    y = pd.read_csv(DataPath/'train_y.csv',dtype='float32')
    if features is not None:
        df = df[features]
    df = df.values
    scaler = StandardScaler()
    df = scaler.fit_transform(df).astype('float32')
    y = y.values
    train_val = [(df[i],y[i]) for i in range(len(df))]
    for thisFile in aug_files:
        print(thisFile)
        df = pd.read_csv(DataPath/thisFile)
        df = df.drop(['seg_id','seg_start','seg_end'],axis=1)
        if features is not None:
            df = df[features]
        df = scaler.transform(df).astype('float32')
        train_val += [(df[i],y[i]) for i in range(len(df))]

    return train_val

def create_result_dir(prefix):
    result_dir = Path(out,'{}_{}'.format(prefix,time.strftime('%Y-%m-%d_%H-%M-%S')))
    (result_dir/'snapshots').mkdir(exist_ok=True,parents=True)
    shutil.copy(__file__, result_dir/Path(__file__).name)

    return result_dir

def main():
    parser = argparse.ArgumentParser(description='Change Point Detection')
    parser.add_argument('--model_path', type=str, default='tyamaguchi/src/models/DNN.py',
                        help='Path of model file')
    parser.add_argument('--model_name', type=str, default='DNN',
                        help='Model class name')
    parser.add_argument('--train_file', '-t', type=str, default='train_x.csv',
                        help='Model class name')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.01,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=500,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()



    ModelPath = Path(args.model_path)
    ModelName = args.model_name
    TrainFile = args.train_file
    gpu_id = args.gpu
    lr = args.learnrate
    batchsize = args.batchsize
    epoch = args.epoch
    n_folds = 6


    result_dir = create_result_dir(ModelName)
    shutil.copy(ModelPath,result_dir/ModelPath.name)
    train_val = make_dataset(TrainFile,top_features,AugFiles)

    datasets = get_cross_validation_datasets(train_val,n_folds)


    for n_fold in range(n_folds):
        train, val = datasets[n_fold]
        loss = lambda x,y:F.mean_absolute_error(x,y) # +F.mean_squared_error(x,y)
        model_path = os.path.splitext(ModelPath)[0].replace('/','.')
        model_module = import_module(model_path)
        model = getattr(model_module, ModelName)()
        model = L.Classifier(model,lossfun=loss,accfun=F.mean_absolute_error)



        if gpu_id >= 0:
            chainer.backends.cuda.get_device_from_id(gpu_id).use()
            model.to_gpu()


        optimizer = chainer.optimizers.Adam(lr)
        # optimizer = chainer.optimizers.MomentumSGD(lr)
        optimizer.setup(model)
        # optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))



        train_iter = chainer.iterators.SerialIterator(train, batchsize)
        val_iter = chainer.iterators.SerialIterator(val, batchsize,
                                                          repeat=False, shuffle=False)

        stop_trigger = training.triggers.EarlyStoppingTrigger(check_trigger=(1, 'epoch'), monitor='validation/main/loss', patients=50, mode='auto', verbose=False, max_trigger=(epoch, 'epoch'))


        updater = training.updaters.StandardUpdater(
            train_iter, optimizer, device=gpu_id)
        trainer = training.Trainer(updater, stop_trigger, out=result_dir)
        trainer.extend(extensions.Evaluator(val_iter, model, device=gpu_id))
        # trainer.extend(extensions.ExponentialShift('lr', 0.5),
        #                trigger=(100, 'epoch'))
        trainer.extend(extensions.ExponentialShift('alpha', 0.5),
                       trigger=(100, 'epoch'))

        # trainer.extend(extensions.snapshot(filename='snaphot_epoch_{.updater.epoch}'),trigger=(epoch, 'epoch'))

        trainer.extend(extensions.LogReport(trigger=(1,'epoch')))

        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

        trainer.extend(extensions.ProgressBar())
        trainer.run()
        serializers.save_npz(result_dir/'snapshots'/'snaphot_epoch_{}_{}'.format(n_fold,trainer.updater.epoch), trainer)

if __name__ == '__main__':
    main()
