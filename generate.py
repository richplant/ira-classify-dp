# Copyright 2019 RBC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# evaluate.py is used to create the synthetic data generation and evaluation pipeline.

from sklearn import preprocessing
from scipy.special import expit
from models import dp_wgan, pate_gan, ron_gauss
import argparse
import numpy as np
import pandas as pd
import collections
import os

parser = argparse.ArgumentParser()
parser.add_argument('--target-variable', required=True, help='Target class')
parser.add_argument('--train-data-path', required=True)
parser.add_argument('--test-data-path', required=True)
parser.add_argument('--normalize-data', action='store_true', help='Apply sigmoid function to each value in the data')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')

privacy_parser = argparse.ArgumentParser(add_help=False)

privacy_parser.add_argument('--enable-privacy', action='store_true', help='Enable private data generation')
privacy_parser.add_argument('--target-epsilon', type=float, default=8, help='Epsilon differential privacy parameter')
privacy_parser.add_argument('--target-delta', type=float, default=1e-5, help='Delta differential privacy parameter')
privacy_parser.add_argument('--output-data-path', required=True, help='Synthetic data save path')

noisy_sgd_parser = argparse.ArgumentParser(add_help=False)

noisy_sgd_parser.add_argument('--sigma', type=float,
                              default=2, help='Gaussian noise variance multiplier. A larger sigma will make the model '
                                              'train for longer epochs for the same privacy budget')
noisy_sgd_parser.add_argument('--clip-coeff', type=float,
                              default=0.1, help='The coefficient to clip the gradients before adding noise for private '
                                                'SGD training')
noisy_sgd_parser.add_argument('--micro-batch-size',
                              type=int, default=8,
                              help='Parameter to tradeoff speed vs efficiency. Gradients are averaged for a microbatch '
                                   'and then clipped before adding noise')

noisy_sgd_parser.add_argument('--num-epochs', type=int, default=500)
noisy_sgd_parser.add_argument('--batch-size', type=int, default=64)

subparsers = parser.add_subparsers(help="generative model type", dest="model")

parser_pate_gan = subparsers.add_parser('pate-gan', parents=[privacy_parser])
parser_pate_gan.add_argument('--lap-scale', type=float,
                             default=0.0001, help='Inverse laplace noise scale multiplier. A larger lap_scale will '
                                                  'reduce the noise that is added per iteration of training.')
parser_pate_gan.add_argument('--batch-size', type=int, default=64)
parser_pate_gan.add_argument('--num-teachers', type=int, default=10, help="Number of teacher disciminators in the pate-gan model")
parser_pate_gan.add_argument('--teacher-iters', type=int, default=5, help="Teacher iterations during training per generator iteration")
parser_pate_gan.add_argument('--student-iters', type=int, default=5, help="Student iterations during training per generator iteration")
parser_pate_gan.add_argument('--num-moments', type=int, default=100, help="Number of higher moments to use for epsilon calculation for pate-gan")

parser_ron_gauss = subparsers.add_parser('ron-gauss', parents=[privacy_parser])

parser_dp_wgan = subparsers.add_parser('dp-wgan', parents=[privacy_parser, noisy_sgd_parser])
parser_dp_wgan.add_argument('--clamp-lower', type=float, default=-0.01, help="Clamp parameter for wasserstein GAN")
parser_dp_wgan.add_argument('--clamp-upper', type=float, default=0.01, help="Clamp parameter for wasserstein GAN")

opt = parser.parse_args()

# Loading the data
train = pd.read_csv(opt.train_data_path)
test = pd.read_csv(opt.test_data_path)

data_columns = [col for col in train.columns if col != opt.target_variable]

class_ratios = train[opt.target_variable].sort_values().groupby(train[opt.target_variable]).size().values/train.shape[0]

X_train = np.nan_to_num(train.drop([opt.target_variable], axis=1).values)
y_train = np.nan_to_num(train[opt.target_variable].values)
X_test = np.nan_to_num(test.drop([opt.target_variable], axis=1).values)
y_test = np.nan_to_num(test[opt.target_variable].values)

if opt.normalize_data:
    X_train = expit(X_train)
    X_test = expit(X_test)

input_dim = X_train.shape[1]
z_dim = int(input_dim / 4 + 1) if input_dim % 4 == 0 else int(input_dim / 4)

# Training the generative model
if opt.model == 'pate-gan':
    Hyperparams = collections.namedtuple(
        'Hyperarams',
        'batch_size num_teacher_iters num_student_iters num_moments lap_scale class_ratios lr')
    Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None)

    model = pate_gan.PATE_GAN(input_dim, z_dim, opt.num_teachers, opt.target_epsilon, opt.target_delta, True)
    model.train(X_train, y_train, Hyperparams(batch_size=opt.batch_size, num_teacher_iters=opt.teacher_iters,
                                              num_student_iters=opt.student_iters, num_moments=opt.num_moments,
                                              lap_scale=opt.lap_scale, class_ratios=class_ratios, lr=1e-4))

elif opt.model == 'dp-wgan':
    Hyperparams = collections.namedtuple(
        'Hyperarams',
        'batch_size micro_batch_size clamp_lower clamp_upper clip_coeff sigma class_ratios lr num_epochs')
    Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None, None, None)

    model = dp_wgan.DP_WGAN(input_dim, z_dim, opt.target_epsilon, opt.target_delta, True)
    model.train(X_train, y_train, Hyperparams(batch_size=opt.batch_size, micro_batch_size=opt.micro_batch_size,
                                              clamp_lower=opt.clamp_lower, clamp_upper=opt.clamp_upper,
                                              clip_coeff=opt.clip_coeff, sigma=opt.sigma, class_ratios=class_ratios, lr=
                                              5e-5, num_epochs=opt.num_epochs), private=opt.enable_privacy)

elif opt.model == 'ron-gauss':
    model = ron_gauss.RONGauss(z_dim, opt.target_epsilon, opt.target_delta, True)

# Generating synthetic data from the trained model
if opt.model == 'ron-gauss':
    X_syn, y_syn, dp_mean_dict = model.generate(X_train, y=y_train)

elif opt.model == 'dp-wgan' or opt.model == 'pate-gan':
    syn_data = model.generate(X_train.shape[0], class_ratios)
    X_syn, y_syn = syn_data[:, :-1], syn_data[:, -1]


if not os.path.isdir(opt.output_data_path):
    raise Exception('Output directory does not exist')

X_syn_df = pd.DataFrame(data=X_syn, columns=data_columns)
y_syn_df = pd.DataFrame(data=y_syn, columns=[opt.target_variable])

syn_df = pd.concat([X_syn_df, y_syn_df], axis=1)
syn_df.to_csv(opt.output_data_path + "/synthetic_data.csv")
print("Saved synthetic data at : ", opt.output_data_path)
