#!/usr/bin/env python

# Copyright 2018 Francis Y. Yan
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

import argparse
import select
import socket
import sys
from collections import deque
from multiprocessing import Pipe, Process
from os import path

import context
import numpy as np
from app import App
from helpers.utils import (ALL_FLAGS, ERR_FLAGS, READ_ERR_FLAGS, READ_FLAGS,
                           WRITE_FLAGS, one_hot, timestamp_ms)
from helpers.utils import  Config
from message import Message
from policy import Policy


from sklearn.externals import joblib

class Sender(object):
    def __init__(self, ip, port):
        self.peer_addr = (ip, port)

        # non-blocking UDP socket and poller
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setblocking(0)  # set socket to non-blocking

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

        # sequence numbers
        self.seq_num = 0
        self.next_ack = 0
        self.inflight = 0

        # check sender health
        self.health_check_ms = None
        self.pre_checked_seq_num = 0
        self.pre_checked_next_ack = 0

        # congestion control policy
        self.policy = None

        # sender's max run time
        self.run_time = None
        self.start_time = None

        # dedicate process for sending pkt
        self.send_queue_out, self.send_queue_in = Pipe(False)
        self.send_process = Process(target=self.__send_process_func)

        # on off app limite
        self.app_limit_open = False

        # whether app data can use up the cwnd
        self.app_limited = False

        # application to generate traffic
        self.app = App(0)

# private
    def __window_is_open(self):
        return self.seq_num - self.next_ack < int(self.policy.cwnd)

    def simple_handshake(self):
        self.sock.setblocking(1)
        self.sock.settimeout(0.5)
        msg = ''
        while msg != 'Hello':
            self.sock.sendto('Hello', self.peer_addr)
            try:
                msg, addr = self.sock.recvfrom(1500)
            except socket.timeout:
                continue

        self.sock.setblocking(0)

    def __send(self):
        msg = Message(self.seq_num, timestamp_ms(), self.policy.bytes_sent,
                      self.policy.ack_recv_ts, self.policy.bytes_acked)
        try:
            self.sock.sendto(msg.to_data(), self.peer_addr)
        except socket.error:
            sys.stderr.write('send error\n')
            return -1

        self.seq_num += 1

        # tell policy that a datagram was sent
        self.policy.data_sent()

        return 0

    def __msend(self, num):
        if num == 0:
            return

        if self.app_limit_open:
            self.app_limited = False

        msg_array = []
        ts = timestamp_ms()
        msg_template = Message(self.seq_num, ts, self.policy.bytes_sent, self.policy.ack_recv_ts, self.policy.bytes_acked)
        for _ in xrange(num):

            # check app limited
            if self.app_limit_open and not self.app.get_app_data():
                self.app_limited = True
                break

            msg_array.append(msg_template.header_to_string())

            self.seq_num += 1
            self.policy.data_sent()

            msg_template.seq_num = self.seq_num
            msg_template.data_sent = self.policy.bytes_sent

        self.send_queue_in.send(msg_array)

        if self.app_limit_open:
            self.policy.app_limited = self.app_limited
        self.policy.inflight = self.seq_num - self.next_ack

    def __send_process_func(self):
        _sending_queue = deque()
        while True:
            if self.send_queue_out.poll(0):
                msg_array = self.send_queue_out.recv()
                _sending_queue.extend(msg_array)
            pre_ts = timestamp_ms()
            while _sending_queue and timestamp_ms() - pre_ts < 10:  # timeout = 10ms
                msg_header = _sending_queue.popleft()
                msg = msg_header + Message.dummy_payload
                try:
                    ret = self.sock.sendto(msg, self.peer_addr)
                    if ret == -1:
                        _sending_queue.appendleft(msg_header)
                        break
                except socket.error:
                    _sending_queue.appendleft(msg_header)
                    break

    def __pacing_send(self):
        c = self.policy.pacing_pkt_number(self.policy.cwnd - (self.seq_num - self.next_ack))
        self.__msend(c)

    def __check_sender_health(self):
        if self.health_check_ms is None:
            self.health_check_ms = timestamp_ms()

        if timestamp_ms() - self.health_check_ms > 10000:  # cool down for 10s
            self.health_check_ms = timestamp_ms()

            if self.pre_checked_seq_num == self.seq_num or self.pre_checked_next_ack == self.next_ack:
                self.pre_checked_seq_num = self.seq_num
                self.pre_checked_next_ack = self.next_ack
                print('send time > 10s')
                return False

            self.pre_checked_seq_num = self.seq_num
            self.pre_checked_next_ack = self.next_ack
        return True

    def __recv(self):
        try:
            msg_str, addr = self.sock.recvfrom(1500)
        except socket.error:
            return -1
        if len(msg_str) < Message.header_size:
            return -1

        ack = Message.parse(msg_str)

        # update next ACK's sequence number to expect
        self.next_ack = max(self.next_ack, ack.seq_num + 1)

        # tell policy that an ack was received
        self.policy.ack_received(ack)

        return 0

    def __run_timeout(self):
        if (self.run_time is None or self.policy.train):
            return False

        if timestamp_ms() - self.start_time > self.run_time:
            return True
        else:
            return False

# public
    def cleanup(self):
        self.sock.close()
        self.send_queue_out.close()
        self.send_queue_in.close()
        self.send_process.terminate()

    def set_policy(self, policy):
        self.policy = policy

    def set_run_time(self, time):
        self.run_time = time  # ms

    def run(self):
        if not self.policy:
            sys.exit('sender\'s policy has not been set')

        self.start_time = timestamp_ms()
        self.send_process.start()
        self.simple_handshake()
        
        if self.app_limit_open:
            self.app.start()

        while not self.policy.stop_sender and not self.__run_timeout():
            if not self.__check_sender_health():
                sys.stderr.write('No send or recv packets for 10 senconds. Exited.\n')
                return -1

            if self.__window_is_open():
                self.poller.modify(self.sock, ALL_FLAGS)
            else:
                self.poller.modify(self.sock, READ_ERR_FLAGS)
            events = self.poller.poll(self.policy.timeout_ms())
            if not events:  # timed out; send one datagram to get rolling
                self.__msend(1)

            for fd, flag in events:
                if flag & ERR_FLAGS:
                    sys.exit('[sender] error returned from poller')

                if flag & READ_FLAGS:
                    self.__recv()

                if flag & WRITE_FLAGS:
                    if self.app_limit_open:
                        self.app.flush_app()
                    if self.__window_is_open():
                        if self.policy.pacing:
                            self.__pacing_send()
                        else:
                            num = int(self.policy.cwnd) - (self.seq_num - self.next_ack)
                            self.__msend(num)
        self.policy.bytes_acked += (self.seq_num - self.next_ack) * Message.total_size 

class DSTreeExecuter(object):
    def __init__(self, state_dim, action_cnt, restore_vars):
        self.aug_state_dim = state_dim + action_cnt
        self.action_cnt = action_cnt
        self.prev_action = action_cnt - 1

        self.model = joblib.load(restore_vars)

    def sample_action(self, state):
        # norm_state = normalize(state)
        norm_state = state

        #
        if Config.action_his:
            one_hot_action = one_hot(self.prev_action, self.action_cnt)
            norm_state = norm_state + one_hot_action

        action = self.model.predict([norm_state])[0]
        #print('{},{},{}'.format(self.prev_action,action,norm_state))
        self.prev_action = action

        return action

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip')
    parser.add_argument('port', type=int)
    parser.add_argument('model', action='store')
    args = parser.parse_args()

    sender = None
    try:
        # dummy policy
        # policy = Policy(False)
        # policy.set_sample_action(lambda state: 2)

        # normal policy
        policy = Policy(False)
        dtree = DSTreeExecuter(state_dim=Policy.state_dim,
                            action_cnt=Policy.action_cnt,
                            restore_vars=args.model)
        policy.set_sample_action(dtree.sample_action)

        sender = Sender(args.ip, args.port)
        sender.set_policy(policy)
        sender.policy.set_env_name('env_60_5ms')
        sender.run()
    except KeyboardInterrupt:
        sys.stderr.write('[sender] stopped\n')
    finally:
        if sender:
            sender.cleanup()


if __name__ == '__main__':
    main()
