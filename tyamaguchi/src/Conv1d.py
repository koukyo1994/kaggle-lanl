import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
import numpy as np
import pandas as pd
from pathlib import Path
import os
import shutil
from importlib import import_module
from tqdm import tqdm
import time


TrainPath = Path('tyamaguchi/src/Conv1d.py')
DataPath = Path('input')
MasterFile = 'train_split_master.csv'
delimiter = ','

ModelPath = Path('tyamaguchi','src','models','CNN1D.py')
ModelName = 'CNN1D'
gpu_id = 0
lr = 0.05
batchsize = 256
epoch = 200
out = Path('tyamaguchi/nn_results')


def make_dataset(val_eq_nums=[15,16]):
    train_file_nums = []
    val_file_nums = []

    with open(DataPath/MasterFile, 'r') as reader:
        header = next(reader)
        for line in reader:
            thisData = line.strip().split(delimiter)
            if thisData[1] != '150000':
                continue
            if thisData[-1]!=thisData[-2]:
                continue
            if int(thisData[-1]) in val_eq_nums:
                val_file_nums.append(thisData[0])
            else:
                train_file_nums.append(thisData[0])

    train = []
    val = []


    for i in tqdm(train_file_nums):
        df = pd.read_csv(DataPath/'train_split'/'train_{}.csv'.format(i))
        train.append((np.array([df['acoustic_data'].values],'float32'),np.array([df['time_to_failure'].values[-1]],'float32')))


    for i in tqdm(val_file_nums):
        df = pd.read_csv(DataPath/'train_split'/'train_{}.csv'.format(i))
        val.append((np.array([df['acoustic_data'].values],'float32'),np.array([df['time_to_failure'].values[-1]],'float32')))


    return train, val

def create_result_dir(prefix: str) -> str:
    result_dir = Path(out,'{}_{}'.format(prefix,time.strftime('%Y-%m-%d_%H-%M-%S')))
    result_dir.mkdir(exist_ok=True,parents=True)
    # shutil.copy(__file__, Path(result_dir, __file__).name))

    return result_dir

def main():
    model_path = os.path.splitext(ModelPath)[0].replace('/','.')
    model_module = import_module(model_path)
    model = getattr(model_module, ModelName)()
    model = L.Classifier(model,lossfun=F.mean_absolute_error)
    model.compute_accuracy = False


    if gpu_id >= 0:
        chainer.backends.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()


    optimizer = chainer.optimizers.MomentumSGD(lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

    train,val = make_dataset()
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    val_iter = chainer.iterators.SerialIterator(val, batchsize,
                                                  repeat=False, shuffle=False)

    stop_trigger = (epoch, 'epoch')
    result_dir = create_result_dir(ModelName)
    shutil.copy(ModelPath,result_dir/ModelPath.name)


    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=gpu_id)
    trainer = training.Trainer(updater, stop_trigger, out=result_dir)

    trainer.extend(extensions.Evaluator(val_iter, model, device=gpu_id))
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))

    trainer.extend(extensions.snapshot(filename='snaphot_epoch_{.updater.epoch}'),trigger=(epoch, 'epoch'))

    trainer.extend(extensions.LogReport(trigger=(1,'iteration')))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())
    trainer.run()

if __name__ == '__main__':
    main()
