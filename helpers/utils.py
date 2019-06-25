#!/usr/bin/env python

# Copyright 2018 Francis Y. Yan, Jestin Ma
# Copyright 2018 Wei Wang, Yiyang Shao (Huawei Technologies)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


import ast
import ConfigParser
from ConfigParser import NoOptionError
import errno
import operator
import os
import select
import socket
import sys
import time
from os import path

import context
import numpy as np

READ_FLAGS = select.POLLIN | select.POLLPRI
WRITE_FLAGS = select.POLLOUT
ERR_FLAGS = select.POLLERR | select.POLLHUP | select.POLLNVAL
READ_ERR_FLAGS = READ_FLAGS | ERR_FLAGS
ALL_FLAGS = READ_FLAGS | WRITE_FLAGS | ERR_FLAGS

DEVNULL = open(os.devnull, 'w')


def format_actions(action_list):
    ret = []

    for action in action_list:
        op = action[0]
        val = float(action[1:])

        if op == '+':
            ret.append((operator.add, val))
        elif op == '-':
            ret.append((operator.sub, val))
        elif op == '*':
            ret.append((operator.mul, val))
        elif op == '/':
            ret.append((operator.div, val))

    return ret


def min_x_max(min_value, x, max_value):
    # return x if min_value < x < max_value
    # else return min_value or max_value
    if min_value < x < max_value:
        return x
    elif x >= max_value:
        return max_value
    else:
        return min_value


def timestamp_ms():
    return int(round(time.time() * 1000))


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_open_port():
    sock = socket.socket(socket.AF_INET)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def check_pid(pid):
    """ Check for the existence of a unix pid """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def one_hot(action, action_cnt):
    ret = [0.0] * action_cnt
    ret[action] = 1.0
    return ret


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def update_ewma(ewma, new_value):
    if ewma is None:
        return float(new_value)
    else:
        return 0.875 * ewma + 0.125 * new_value


class RingBuffer(object):
    def __init__(self, length):
        self.full_len = length
        self.real_len = 0
        self.index = 0
        self.data = np.zeros(length)

    def append(self, x):
        self.data[self.index] = x
        self.index = (self.index + 1) % self.full_len
        if self.real_len < self.full_len:
            self.real_len += 1

    def get(self):
        idx = (self.index - self.real_len +
               np.arange(self.real_len)) % self.full_len
        return self.data[idx]

    def reset(self):
        self.real_len = 0
        self.index = 0
        self.data.fill(0)


class MeanVarHistory(object):
    def __init__(self):
        self.length = 0
        self.mean = 0.0
        self.square_mean = 0.0
        self.var = 0.0

    def append(self, x):
        """Append x to history.

        Args:
            x: a list or numpy array.
        """
        # x: a list or numpy array
        length_new = self.length + len(x)
        ratio_old = float(self.length) / length_new
        ratio_new = float(len(x)) / length_new

        self.length = length_new
        self.mean = self.mean * ratio_old + np.mean(x) * ratio_new
        self.square_mean = (self.square_mean * ratio_old +
                            np.mean(np.square(x)) * ratio_new)
        self.var = self.square_mean - np.square(self.mean)

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.var if self.var > 0 else 1e-10

    def get_std(self):
        return np.sqrt(self.get_var())

    def normalize_copy(self, x):
        """Normalize x and returns a copy.

        Args:
            x: a list or numpy array.
        """
        return [(v - self.mean) / self.get_std() for v in x]

    def normalize_inplace(self, x):
        """Normalize x in place.

        Args:
            x: a numpy array with float dtype.
        """
        x -= self.mean
        x /= self.get_std()

    def reset(self):
        self.length = 0
        self.mean = 0.0
        self.square_mean = 0.0
        self.var = 0.0


def ssh_cmd(host):
    return ['ssh', '-q', '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=5', host]


def convert_to_seconds(time_str):
    """ parse time end with different units and format to seconds """

    if time_str.endswith('ms'):
        return float(time_str[:-2]) / 1000.0
    elif time_str.endswith('s'):
        return float(time_str[:-1])
    else:  # default unit: seconds
        return float(time_str)


def parse_trace_file():
    # TODO
    pass


class Config(object):
    cfg = ConfigParser.ConfigParser()
    cfg_path = path.join(context.base_dir, 'config.ini')
    cfg.read(cfg_path)

    # lstm model
    lstm_layer = int(cfg.get('global', 'lstm_layer'))
    lstm_units = int(cfg.get('global', 'lstm_units'))

    # device for train
    device = cfg.get('global', 'device')
    batch_size = int(cfg.get('global', 'batch_size'))

    # model input state index
    state_idx = int(cfg.get('global', 'state_idx'))
    # model input dim number
    state_dim = int(cfg.get('global', 'state_dim'))
    state_his = int(cfg.get('global', 'state_his'))

    # oracle decision boundary
    oracle_his = int(cfg.get('global', 'oracle_his'))

    # action frequency
    action_frequency = float(cfg.get('global', 'action_frequency'))

    # friendliness for CC expert algorithm
    fri = float(cfg.get('global', 'fri'))
    # weight for hard_target and soft_target
    rho = float(cfg.get('global', 'rho'))
    # shuffle window size
    shuffle_window = int(cfg.get('global', 'shuffle_window'))

    # whether launch perf server and client in test mode
    perf = ast.literal_eval(cfg.get('global', 'measurement'))

    # sender run_time in test mode
    run_time = int(cfg.get('global', 'run_time'))


    try:
        rand_rate = float(cfg.get('global', 'rand_rate'))
    except NoOptionError:
        rand_rate = None
    try:
        log_dir = cfg.get('global', 'log_dir')
    except NoOptionError:
        log_dir = None
    try:
        action_his = ast.literal_eval(cfg.get('global', 'action_his'))
    except NoOptionError:
        action_his = True
    try:
        cwnd_ewma = float(cfg.get('global', 'cwnd_ewma'))
    except NoOptionError:
        cwnd_ewma = 0.0
    try:
        auto_ewma = ast.literal_eval(cfg.get('global', 'auto_ewma'))
    except NoOptionError:
        auto_ewma = False
    try:
        ewma_map = ast.literal_eval(cfg.get('global', 'ewma_map'))
    except NoOptionError:
        ewma_map = False
    try:
        diff_his =  ast.literal_eval(cfg.get('global', 'diff_his'))
    except NoOptionError:
        diff_his = False

    try:
        norm_his =  ast.literal_eval(cfg.get('global', 'norm_his'))
    except NoOptionError:
        norm_his = False

    try:
        st_diff_type = int(cfg.get('global', 'st_diff'))
    except NoOptionError:
        st_diff_type = 0

    try:
        st_norm_type = int(cfg.get('global', 'st_norm'))
    except NoOptionError:
        st_norm_type = 0
    try:
        norm_cwnd = ast.literal_eval(cfg.get('global', 'norm_cwnd'))
    except NoOptionError:
        norm_cwnd = False

    try:
        slow_start = ast.literal_eval(cfg.get('global', 'slow_start'))
    except NoOptionError:
        slow_start = False

    try:
        factor = int(cfg.get('global', 'factor'))
    except NoOptionError:
        factor = 1
    
    try:
        auto_factor = ast.literal_eval(cfg.get('global', 'auto_factor'))
    except NoOptionError:
        auto_factor = False

    try:
        state_map =  ast.literal_eval(cfg.get('global', 'st_map'))
    except NoOptionError:
        state_map = {
        'rtt':True,
        'delay':False,
        'send_rate':True,
        'delivery_rate':True,
        'loss_rate':True,
        'cwnd':True
        }
    try:
        his_range =  ast.literal_eval(cfg.get('global', 'his_range'))
    except NoOptionError:
        his_range = [x for x in range(0,state_his)]
    # env (mininet) parameter set and traffic pattern for train mode
    total_env_set_train = []
    total_tpg_set_train = []
    total_env_set_train_name = []
    total_tpg_set_train_name = []
    total_tpg_num_train = 0
    try:
        train_env = cfg.options('train')
        for opt in train_env:
            env_name, tpg_name = ast.literal_eval(cfg.get('train', opt))

            env_param = ast.literal_eval(cfg.get('env', env_name))
            tpg_param = ast.literal_eval(cfg.get('generator', tpg_name))
            gen_name_set = []

            for idx, gen_name in enumerate(tpg_param):
                try:
                    gen_param = ast.literal_eval(cfg.get('generator', gen_name))
                    sketch_name, cycle_str = gen_param
                    sketch_param = ast.literal_eval(cfg.get('generator', sketch_name))
                    cycle_param = convert_to_seconds(cycle_str)
                    tpg_param[idx] = [sketch_param, cycle_param]
                except ValueError:
                    gen_param = cfg.get('generator', gen_name)
                    if type(gen_param) is str and gen_param.endswith('trace'):
                        tpg_param[idx] = parse_trace_file(gen_param)
                    else:
                        tpg_param[idx] = gen_param
                finally:
                    gen_name_set.append(gen_name)

            total_env_set_train.append(env_param)
            total_tpg_set_train.append(tpg_param)
            total_env_set_train_name.append(env_name)
            total_tpg_set_train_name.append(gen_name_set)

        for tp in total_tpg_set_train:
            total_tpg_num_train += len(tp)
    except SyntaxError:
        sys.exit('config error while parsing train/env/generator')

    # env (mininet) parameter set and traffic pattern for test mode
    total_env_set_test = []
    total_tpg_set_test = []
    total_env_set_test_name = []
    total_tpg_set_test_name = []
    try:
        test_env = cfg.options('test')
        for opt in test_env:
            env_name, tpg_name = ast.literal_eval(cfg.get('test', opt))

            env_param = ast.literal_eval(cfg.get('env', env_name))
            tpg_param = ast.literal_eval(cfg.get('generator', tpg_name))
            gen_name_set = []

            for idx, gen_name in enumerate(tpg_param):
                try:
                    gen_param = ast.literal_eval(cfg.get('generator', gen_name))
                    sketch_name, cycle_str = gen_param
                    sketch_param = ast.literal_eval(cfg.get('generator', sketch_name))
                    cycle_param = convert_to_seconds(cycle_str)
                    tpg_param[idx] = [sketch_param, cycle_param]
                except ValueError:
                    gen_param = cfg.get('generator', gen_name)
                    if type(gen_param) is str and gen_param.endswith('trace'):
                        tpg_param[idx] = parse_trace_file(gen_param)
                    else:
                        tpg_param[idx] = gen_param
                finally:
                    gen_name_set.append(gen_name)

            total_env_set_test.append(env_param)
            total_tpg_set_test.append(tpg_param)
            total_env_set_test_name.append(env_name)
            total_tpg_set_test_name.append(gen_name_set)
    except SyntaxError:
        sys.exit('config error while parsing test/env/generator')

    if perf :
        try:
            data_dir_prefix = cfg.get('statistics', 'data_dir_prefix')
        except SyntaxError:
            sys.exit('config error while parsing test/statistics/data_dir_prefix')
