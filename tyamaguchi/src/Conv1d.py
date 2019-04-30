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
import argparse
import sys
from statsmodels.robust import mad
import scipy
from scipy import signal
from scipy.signal import butter
import pywt

sys.path.append('.')

DataPath = Path('input/train_wave_split')
delimiter = ','
out = Path('tyamaguchi/nn_results')

rows = 150_000
n_samples = 150_000
sample_duration = 0.02
sample_rate = n_samples * (1 / sample_duration)

def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def high_pass_filter(x, low_cutoff=1000, sample_rate=sample_rate):
    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist
    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = signal.sosfilt(sos, x)

    return filtered_sig

def denoise_signal( x, wavelet='db4', level=1):
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    sigma = (1/0.6745) * maddest( coeff[-level] )
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard' ) for i in coeff[1:] )

    return pywt.waverec( coeff, wavelet, mode='per' )

def make_dataset(slide=150_000, val_eq_nums=[1,15,16]):
    train = []
    val = []

    for eq_num in tqdm(range(17)):
        data = pd.read_csv(DataPath/'train_wave_{}.csv'.format(eq_num))
        acoustic_data = data['acoustic_data'].values
        time_data = data['time_to_failure'].values
        for n in tqdm(range((len(data)-rows)//slide+1)):
            x = acoustic_data[slide*n:slide*n+rows]
            y = time_data[slide*n+rows-1]
            x = high_pass_filter(x, low_cutoff=10000, sample_rate=sample_rate)
            x = denoise_signal(x, wavelet='haar', level=1)
            x = F.max_pooling_1d(np.array([[x]]),100,100)
            x = x[0][0].data
            if eq_num not in val_eq_nums:
                train.append((np.array([x],'float32'),(np.array([y],'float32'))))
            else:
                val.append((np.array([x],'float32'),(np.array([y],'float32'))))

    return train, val

def create_result_dir(prefix):
    result_dir = Path(out,'{}_{}'.format(prefix,time.strftime('%Y-%m-%d_%H-%M-%S')))
    result_dir.mkdir(exist_ok=True,parents=True)
    shutil.copy(__file__, result_dir/Path(__file__).name)

    return result_dir

def main():
    parser = argparse.ArgumentParser(description='Change Point Detection')
    parser.add_argument('--model_path', type=str, default='tyamaguchi/src/models/CNN1D.py',
                        help='Path of model file')
    parser.add_argument('--model_name', type=str, default='CNN1D',
                        help='Model class name')
    parser.add_argument('--slide', '-s', type=int, default=150_000,
                        help='Length of sliding window')
    parser.add_argument('--batchsize', '-b', type=int, default=256,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()



    # ModelPath = Path('tyamaguchi','src','models','CNN1D.py')
    # ModelName = 'CNN1D'
    # gpu_id = 0
    # lr = 0.05
    # batchsize = 256
    # epoch = 300
    # out = Path('tyamaguchi/nn_results')

    ModelPath = Path(args.model_path)
    ModelName = args.model_name
    slide = args.slide
    gpu_id = args.gpu
    lr = args.learnrate
    batchsize = args.batchsize
    epoch = args.epoch


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

    train,val = make_dataset(slide)
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

    trainer.extend(extensions.LogReport(trigger=(1,'epoch')))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())
    trainer.run()

if __name__ == '__main__':
    main()
