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


import collections
import datetime
import random
import sys
import math
import context  # noqa # pylint: disable=unused-import
from helpers.utils import Config, format_actions, timestamp_ms, update_ewma,timestamp_us
from message import Message


class Policy(object):
    min_cwnd = 2.0
    max_cwnd = 25000.0

    max_rtt = 300.0  # ms
    max_delay = max_rtt
    max_send_rate = 1000.0  # Mbps
    max_delivery_rate = max_send_rate

    min_step_len = Config.mcall_step_len  # ms
    steps_per_episode = 1000  # number of steps in each episode (in training)

    # state = [rtt_norm, delay_norm, send_rate_norm, delivery_rate_norm,
    #          loss_rate_norm, cwnd_norm]
    state_dim = Config.state_dim * Config.state_his
    label_dim = 3  # len([cwnd, expert_cwnd, expert_action])
    action_list = ["/1.5", "/1.05", "+0.0", "*1.05", "*1.5"]
    action_cnt = len(action_list)
    action_mapping = format_actions(action_list)
    action_frequency = Config.action_frequency

    delay_ack = True
    delay_ack_count = 2
    state_map = {
        'rtt':False,
        'delay':True,
        'send_rate':False,
        'delivery_rate':False,
        'loss_rate':False,
        'cwnd':True
    }
    s_names = ['rtt','delay','send_rate','delivery_rate','loss_rate','cwnd']
    c_ewma_k = 0.375 / math.log(30) 
    ewma_tab = [ 0.0,  0.0,  0.0,  0.0,  0.125 , 0.125, 0.125, 0.125, 0.125, 0.125, #0~50ms
                 0.25, 0.25, 0.25, 0.25, 0.25,   0.25,  0.25,  0.25,  0.25,  0.25, #50~100ms
                 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375, 0.375 #100~150ms
               ]
    #c_ewma_k =  0.375/ math.sqrt(145)
    def __init__(self, train):
        # public:
        self.cwnd = 10.0
        self.bytes_sent = 0
        #timestamp of lastest recieved ack 
        self.ack_recv_ts = 0
        #total bytes to be recieved 
        self.bytes_acked = 0

        # sender should stop or not
        self.stop_sender = False

        # pacing or not
        self.pacing = True

        self.app_limited = False
        self.inflight = 0

    # private:
        self.train = train
        self.sample_action = None

        # step timer and counting
        self.step_start_ts = None
        self.step_num = 0
        self.start_phase_cnt = 0
        self.start_phase_max = 4  # max number of steps in start phase

        # state related (persistent across steps)
        self.min_rtt = sys.maxint
        self.max_rtt = 0
        self.min_delay_ewma = float('inf')
        self.max_delay_ewma = 0.0
        self.min_send_rate_ewma = float('inf')
        self.max_send_rate_ewma = 0.0
        self.min_delivery_rate_ewma = float('inf')
        self.max_delivery_rate_ewma = 0.0

        # variables to calculate loss rate (persistent across steps)
        #sytes_acked at prev step 
        self.prev_bytes_acked = 0
        self.prev_bytes_sent_in_ack = 0
        self.bytes_sent_in_ack = 0

        # state related (reset at each step)
        self.rtt_ewma = None
        self.delay_ewma = None
        self.send_rate_ewma = None
        self.delivery_rate_ewma = None
        self.history_state = collections.deque()

        # pacing related
        self.borrowed_pkt = 0.0
        self.interval = 10.0  # us
        self.pre_sent_ts = None

        # measurement client n in test mode
        self.perf_client = None

        # rtt re-update
        self.rtt_reupdate_interval = 50  # RTT_ewma
        self.rtt_reupdate_start_ts = None

        #last one action
        self.action = None
        self.prev_action = None
        self.prev_cwnd = 0
        self.loss_rate = 0
        self.rand_rate = 0 if Config.rand_rate is None or self.train is False else   Config.rand_rate
        self.env_name = None
        self.init_rate = 1.0
        self.slow_start = Config.slow_start
        self.slow_iter = 0
        self.max_bdp_norm = 0.0 
        self.history_delay = collections.deque()
        self.history_loss = collections.deque()
        self.history_avg_delay = 0.0
        self.history_avg_loss = 0.0
        self.exp_seq = 1
        self.f_loss = False
        self.freeze_cwnd = 2.0
        self.m_timestamp = timestamp_us()
        self.d_stage =True
        self.loss_num =0
        #self.rand_rate = 0.05
        print('policy config action rand_rate {}'.format(self.rand_rate))
        self.cwnd_ewma = Config.cwnd_ewma
        print('policy config cwnd ewma {}'.format(self.cwnd_ewma ))
        print('policy config auto ewma {}'.format(Config.auto_ewma ))
        print('policy config state_idx {}'.format(Config.state_idx))
        print('policy config his_range {}'.format(Config.his_range))
        print('policy config slow start {}'.format(Config.slow_start))
        if Config.diff_his:
            print('policy config diff_his {}'.format(Config.diff_his))
            print('policy config st_diff_type {}'.format(Config.st_diff_type))
        
        print('policy config norm_cwnd {}'.format(Config.norm_cwnd ))
        if Config.norm_his:
            print('policy config norm_his {}'.format(Config.norm_his))
            print('policy config st_norm_type {}'.format(Config.st_norm_type))
            print('policy config factor {}'.format(Config.factor ))
            print('policy config auto_factor {}'.format(Config.auto_factor ))
        if 0 in Config.his_range:
             print('policy config stmap {}'.format(Config.state_map))
        print('policy config Mess size {}'.format(Message.total_size))
        print('policy config calling step time {}ms'.format(Policy.min_step_len))
# private
    def __update_state(self, ack):
        # update RTT and queuing delay (in ms)
        rtt = max(1, self.ack_recv_ts - ack.send_ts)
        if rtt < self.min_rtt and (Config.auto_ewma  or Config.ewma_map):
            if Config.ewma_map:
                cwnd_ewma = Policy.ewma_tab[min(int(rtt/10),30)]
                #cwnd_ewma = Policy.c_ewma_k*  math.sqrt (rtt/2 - 5)
            else:
                rtt_r = max(1.0,rtt/10.0)
                cwnd_ewma = Policy.c_ewma_k * math.log(rtt_r)
                
            print('ewma update {}'.format(cwnd_ewma))

        self.min_rtt = min(self.min_rtt, rtt)
        self.max_rtt = max(self.max_rtt, rtt)
        self.rtt_ewma = update_ewma(self.rtt_ewma, rtt)

        queuing_delay = rtt - self.min_rtt
        self.delay_ewma = update_ewma(self.delay_ewma, queuing_delay)

        self.min_delay_ewma = min(self.min_delay_ewma, self.delay_ewma)
        self.max_delay_ewma = max(self.max_delay_ewma, self.delay_ewma)

        # update sending rate (in Mbps)
        send_rate = 0.008 * (self.bytes_sent - ack.bytes_sent) / rtt
        self.send_rate_ewma = update_ewma(self.send_rate_ewma, send_rate)

        self.min_send_rate_ewma = min(self.min_send_rate_ewma,
                                      self.send_rate_ewma)
        self.max_send_rate_ewma = max(self.max_send_rate_ewma,
                                      self.send_rate_ewma)

        # update delivery rate (in Mbps)
        duration = max(1, self.ack_recv_ts - ack.ack_recv_ts)
        delivery_rate = 0.008 * (self.bytes_acked - ack.bytes_acked) / duration
        self.delivery_rate_ewma = update_ewma(self.delivery_rate_ewma,
                                              delivery_rate)

        self.min_delivery_rate_ewma = min(self.min_delivery_rate_ewma,
                                          self.delivery_rate_ewma)
        self.max_delivery_rate_ewma = max(self.max_delivery_rate_ewma,
                                          self.delivery_rate_ewma)

        # record the sent bytes in the current ACK
        self.bytes_sent_in_ack = ack.bytes_sent


        if self.exp_seq + 2 < ack.seq_num:
            #sys.stderr.write('[{}]detect pakcet loss at {}-{}={}\n'.format(self.loss_num,self.exp_seq,ack.seq_num,ack.seq_num-self.exp_seq))
            self.loss_num +=1
            self.f_loss = True
        #loss_rate = 1.0 if self.exp_seq + 2 < ack.seq_num else 0.0 
        S_loss =self.__cal_loss_rate()
        loss_rate = 1.0 if S_loss>0 else 0.0  
        
        left_delay = 0
        #s_loss =self.__cal_loss_rate()
        if len(self.history_delay) == 0:
            for _ in xrange(10):
                self.history_delay.append(queuing_delay)
                self.history_avg_delay = queuing_delay
                left_delay = 0
        elif len(self.history_delay) == 10:
            left_delay = self.history_delay.popleft()
            self.history_delay.append(queuing_delay) 
            self.history_avg_delay = self.history_avg_delay - left_delay/10.0 + queuing_delay/10.0
            #print('pop left {} ,put right {} '.format(left_delay,queuing_delay))
        if len(self.history_loss) == 0:
            for _ in xrange(20):
                self.history_loss.append(0.0)
                self.history_avg_loss = 0.0
        elif len(self.history_loss) == 20:
            left_loss = self.history_loss.popleft()
            self.history_loss.append(loss_rate) 
            self.history_avg_loss = self.history_avg_loss - left_loss/20.0 + loss_rate/20.0

        if self.slow_start :
            self.cwnd += 2 if self.d_stage or self.cwnd < self.freeze_cwnd else 2*0.3 
            #self.cwnd += 2
            #if self.f_loss:
            #sys.stderr.write('iter[{}]:seq {},ack {}, cwnd {:.2f},fr_cwnd {:.2f},min_rtt {:.2f},send {:.2f},delivery {:.2f},rtt {:.2f},q_delay {:.2f},avg_delay {:.2f},loss {:.6f},avg_loss {:.2f}\n'.format(self.slow_iter,
            #        self.sender.seq_num,ack.seq_num,self.cwnd, self.freeze_cwnd ,self.min_rtt,
            #        self.send_rate_ewma,self.delivery_rate_ewma,self.rtt_ewma,queuing_delay,self.history_avg_delay,S_loss,self.history_avg_loss))  
               # self.f_loss = False
             
        #elif not self.first_loss:
        #    print('iter[{}]:seq {},ack {}, cwnd {:.4f},min_rtt {:.4f},send {:.4f},delivery {:.4f},delay_ewma {:.4f},q_delay {:.4f},avg_delay {:.4f}'.format(self.slow_iter,
        #        self.sender.seq_num,ack.seq_num,self.cwnd,self.min_rtt,
        #        self.send_rate_ewma,self.delivery_rate_ewma,self.delay_ewma,queuing_delay,self.history_avg_delay))              

        #break up slow start
        if self.slow_start and  (self.exp_seq + 4 < ack.seq_num or  S_loss > 0.5 )and self.history_avg_delay / self.min_rtt > 0.01:
            self.slow_start = False
            if self.d_stage:
                max_cwnd = self.cwnd/2  
            else:
                max_cwnd = max(self.freeze_cwnd, (self.cwnd - self.freeze_cwnd)*Config.fri +  self.freeze_cwnd)
            #max_cwnd = self.cwnd /2
            self.max_bdp_norm = max_cwnd*Message.total_size*8/(Policy.max_rtt*Policy.max_delivery_rate*10**3)
            #sys.stderr.write('iter[{}]:seq {},ack {}, cwnd {:.4f},fr_cwnd {:.4f},min_rtt {:.4f},send {:.4f},delivery {:.4f},rtt {:.4f},q_delay {:.4f},avg_delay {:.4f},loss {:.6f},avg_loss {:.4f}\n'.format(self.slow_iter,
            #    self.sender.seq_num,ack.seq_num,self.cwnd,self.freeze_cwnd,self.min_rtt,
            #    self.send_rate_ewma,self.delivery_rate_ewma,self.rtt_ewma,queuing_delay,self.history_avg_delay,S_loss,self.history_avg_loss))  
            #sys.stderr.write('slow start compute max_cwnd {},max_bdp_norm {}\n'.format(max_cwnd,self.max_bdp_norm))
            self.cwnd = self.freeze_cwnd
        #slow start congestion recvory
        elif self.slow_start and self.d_stage and self.history_avg_delay / self.min_rtt > 0.05:
            self.d_stage = False
            self.freeze_cwnd = self.cwnd/2
            self.cwnd = self.freeze_cwnd
            #sys.stderr.write('iter[{}]:seq {},ack {}, cwnd {:.4f},fr_cwnd {:.4f},min_rtt {:.4f},send {:.4f},delivery {:.4f},rtt {:.4f},q_delay {:.4f},avg_delay {:.4f},loss {:.6f},avg_loss {:.4f}\n'.format(self.slow_iter,
            #    self.sender.seq_num,ack.seq_num,self.cwnd, self.freeze_cwnd ,self.min_rtt,
            #    self.send_rate_ewma,self.delivery_rate_ewma,self.rtt_ewma,queuing_delay,self.history_avg_delay,S_loss,self.history_avg_loss))  
            #sys.stderr.write('slow start compute freeze_cwnd {}\n'.format(self.freeze_cwnd))
        self.exp_seq = ack.seq_num

        if not self.slow_start and Config.slow_start:
            try_bdp_norm = (self.min_rtt*self.delivery_rate_ewma*1.1)/(Policy.max_rtt*Policy.max_delivery_rate)
            if try_bdp_norm > self.max_bdp_norm:
                self.max_bdp_norm = try_bdp_norm
                #print('slow start update max_bdp_norm {}'.format(self.max_bdp_norm))
    # calculate loss rate at the end of each step
    # step_acked = acked bytes during this step
    # step_sent = sent bytes recorded in ACKs received at this step
    # loss = 1 - step_acked / step_sent
    def __cal_loss_rate(self):
        step_sent = self.bytes_sent_in_ack - self.prev_bytes_sent_in_ack
        if step_sent == 0:  # prevent divide-by-0 error
            return 0

        step_acked = self.bytes_acked - self.prev_bytes_acked
        loss_rate = 1.0 - float(step_acked) / step_sent

        self.prev_bytes_acked = self.bytes_acked
        self.prev_bytes_sent_in_ack = self.bytes_sent_in_ack

        # in case packet reordering occurred
        return min(max(0.0, loss_rate), 1.0)

    def __take_action(self, action):
        if action < 0 or action >= Policy.action_cnt:
            sys.exit('invalid action')

        op, val = Policy.action_mapping[action]
        if self.train:
            self.cwnd = op(self.cwnd, val)
            self.cwnd = max(Policy.min_cwnd, min(Policy.max_cwnd, self.cwnd))
        else:
            tmp_cwnd = op(self.cwnd, val)
            tmp_cwnd = max(Policy.min_cwnd, min(Policy.max_cwnd, tmp_cwnd))
            cwnd_ewma = Policy.ewma_tab[int(self.min_rtt/10)] if Config.ewma_map else Policy.c_ewma_k * math.log(max(1.0,self.min_rtt/10.0)) if Config.auto_ewma  else self.cwnd_ewma
            #print('cwnd_ewma{}'.format(cwnd_ewma))
            #cwnd_ewma = Policy.c_ewma_k *  math.sqrt (self.min_rtt/2 -5)
            self.cwnd = cwnd_ewma * self.cwnd + (1-cwnd_ewma) * tmp_cwnd

    # reset some stats at each step
    def __reset_step(self):
        self.rtt_ewma = None
        self.delay_ewma = None
        self.send_rate_ewma = None
        self.delivery_rate_ewma = None

    def __episode_ended(self):
        self.stop_sender = True

    def __step_ended(self):
        if self.stop_sender:
            return
        # normalization
        rtt_norm = self.rtt_ewma / Policy.max_rtt
        delay_norm = self.delay_ewma / Policy.max_rtt
        max_rtt_norm = self.max_rtt / Policy.max_rtt
        min_rtt_norm = self.min_rtt / Policy.max_rtt
        send_rate_norm = self.send_rate_ewma / Policy.max_send_rate
        delivery_rate_norm = self.delivery_rate_ewma / Policy.max_delivery_rate
        cwnd_norm = self.cwnd / Policy.max_cwnd
        loss_rate_norm = self.__cal_loss_rate()  # loss is already in [0, 1]
        self.loss_rate = loss_rate_norm

        if self.slow_start and not self.train:
            self.slow_iter +=1
            if self.perf_client:
                self.perf_client.collect_perf_data(self)
                self.prev_cwnd = self.cwnd
            self.__reset_step()
            return 

        # state -> action
        state_array = [[rtt_norm, delay_norm, send_rate_norm, delivery_rate_norm,
                        loss_rate_norm, cwnd_norm],
                       [rtt_norm, delay_norm, max_rtt_norm, min_rtt_norm, send_rate_norm, delivery_rate_norm,
                        loss_rate_norm, cwnd_norm],
                        [rtt_norm, delay_norm, send_rate_norm, delivery_rate_norm,
                        loss_rate_norm]
                        ]
        state = state_array[Config.state_idx]

        # print state,self.min_rtt,self.max_rtt
 
        
        if len(self.history_state) == 0:
            for _ in xrange((Config.state_his -1)*30 + 1):
                self.history_state.append(state)
        elif len(self.history_state) == (Config.state_his -1)*30 + 1:
            self.history_state.popleft()
            self.history_state.append(state) 

        h_state = []
#        for s in self.history_state:
#            h_state = h_state + s
        if 0 in Config.his_range:
            h_state += reduce(lambda s,(x,v): s + [v if not Config.norm_cwnd  else 
                                                    v*(Message.total_size+20+14+20+8)/(self.max_bdp_norm) if x == 'cwnd' else 
                                                        v/min_rtt_norm if x =='rtt' or x =='delay' else
                                                            v/(self.max_bdp_norm/min_rtt_norm) if x=='send_rate' or x=='delivery_rate' else v] 
                                                if Config.state_map[x] else s,zip(Policy.s_names,state),[])
      
        #print('cwnd {},norm_cwnd {}'.format(state[5],h_state[1]))
        
        #if Config.state_his > 1:
        #    r_delay = (state[0] - min_rtt_norm)/min_rtt_norm
        #    h_state +=[r_delay]
        sch_interval = max(self.min_rtt / Policy.action_frequency, Policy.min_step_len)
        factor = int(self.min_rtt/sch_interval) if Config.auto_factor else Config.factor
        factor = 1 if factor == 0 else 30 if factor > 30 else  factor
        #print('factor {}'.format(factor))
        #factor = factor / (Config.state_his - 1) if  Config.state_his > 1 else factor
        #factor = 1
        #ratio = 0
        #print('factor {}'.format(factor))
        #print('bdp_norm {}'.format(self.max_bdp_norm))
        for i in range(1,Config.state_his):
            if i not in Config.his_range:
                continue
            base = len(self.history_state) -1
            if Config.norm_his:
                if Config.st_norm_type ==6 :
                    data1 = self.history_state[base]
                    data0 = self.history_state[base - factor*i]
                
                    n_delay = (data1[1] - data0[1])/min_rtt_norm
                    n_loss = (data1[4] - data0[4]) / data0[4] if data0[4] != 0.0 else 0 
                    n_cwnd = (data1[5] - data0[5])*(Message.total_size+20+14+20+8)/(self.max_bdp_norm)
                    n_send = (data1[2] - data0[2]) /(self.max_bdp_norm/min_rtt_norm)
                    n_delivery  = (data1[3] - data0[3]) /(self.max_bdp_norm/min_rtt_norm)
                    h_state += [n_delay,n_send,n_delivery,n_loss,n_cwnd]
                elif  Config.st_norm_type == 1:
                    data1 = self.history_state[base]
                    data0 = self.history_state[base - factor*i]
                    
                    n_delay = (data1[1] - data0[1])/data0[1]  if data0[1]!=0 else 0.0
                    n_loss = (data1[4] - data0[4]) / data0[4] if data0[4] != 0.0 else 0 
                    n_cwnd = (data1[5] - data0[5])*(Message.total_size+20+14+20+8)/(self.max_bdp_norm)
                    n_send = (data1[2] - data0[2]) /(self.max_bdp_norm/min_rtt_norm)
                    n_delivery  = (data1[3] - data0[3]) /(self.max_bdp_norm/min_rtt_norm)
                    h_state += [n_delay,n_send,n_delivery,n_loss,n_cwnd]
                elif  Config.st_norm_type == 7:
                    data1 = self.history_state[base]
                    data0 = self.history_state[base - factor*i]
                    
                    n_delay = (data1[1] - data0[1])/data0[1]  if data0[1]!=0 else 0.0
                    n_loss = (data1[4] - data0[4]) / data0[4] if data0[4] != 0.0 else 0 
                    n_cwnd = (data1[5] - data0[5])*(Message.total_size+20+14+20+8)/data0[5]
                    n_send = (data1[2] - data0[2]) /data0[2]
                    n_delivery  = (data1[3] - data0[3]) /data0[3]
                    h_state += [n_delay,n_send,n_delivery,n_loss,n_cwnd]
                elif  Config.st_norm_type == 2:
                    n_delay = (data1[0] - min_rtt_norm)/min_rtt_norm
                    n_loss = (data1[4] - data0[4]) / data0[4] if data0[4] != 0.0 else 0 
                    n_cwnd = (data1[5] - data0[5]) / data0[5] 
                    n_delta_rate1 = (data1[3] - data0[2]) /  data0[2]
                    h_state += [n_delay,n_delta_rate1,n_loss,n_cwnd]
                elif Config.st_norm_type == 3:
                    n_rtt = (data1[0] - data0[0])/ data0[0]
                    n_delay = (data1[1] - data0[1])/ data0[1] if data0[1]!=0 else 1.0
                    n_loss = (data1[4] - data0[4]) / data0[4] if data0[4] != 0.0 else 0 
                    n_cwnd = (data1[5] - data0[5]) / data0[5] 
                    n_send = (data1[2] - data0[2]) / data0[2]
                    n_delivery = (data1[3] - data0[3]) / data0[3]
                    h_state += [n_rtt,n_delay,n_send,n_delivery,n_loss,n_cwnd]
                elif Config.st_norm_type == 4:
                    n_delay = (data1[1] - data0[1])/ data0[1] if data0[1]!=0 else 1.0
                    n_loss = (data1[4] - data0[4]) / data0[4] if data0[4] != 0.0 else 0 
                    n_cwnd = (data1[5] - data0[5]) / data0[5] 
                    n_send = (data1[2] - data0[2]) / data0[2]
                    n_delivery = (data1[3] - data0[3]) / data0[3]
                    h_state += [n_delay,n_send,n_delivery,n_loss,n_cwnd]
                elif Config.st_norm_type ==5 :
                    n_delay = (data1[1] - data0[1])/ data0[1] if data0[1]!=0.0 else 0.0
                    n_loss = (data1[4] - data0[4]) / data0[4] if data0[4] != 0.0 else 0 
                    n_cwnd = (data1[5] - data0[5]) / data0[5] 
                    n_send = (data1[2] - data0[2]) / data0[2]
                    n_delivery = (data1[3] - data0[3]) / data0[3]
                    h_state += [n_delay,n_send,n_delivery,n_loss,n_cwnd]

            elif Config.diff_his:
                data0 = self.history_state[base - i]
                data1 = state
                if Config.st_diff_type == 0:
                    diff_state = map(lambda (x0,x1):(x1-x0),zip(data0,data1))
                    h_state += diff_state
                elif Config.st_diff_type == 1:
                    diff_state = map(lambda (x0,x1):(x1-x0)/i,zip(data0,data1))
                    h_state += diff_state                    
                else:
                    diff_state = map(lambda (x0,x1):(x1-x0)/x0 if x0 !=0 else 1.0 ,zip(data0,data1))
                    h_state += diff_state                                
            else:
                h_state +=  self.history_state[base -i]

        if self.sample_action is None:
            sys.exit('sample_action on policy has not been set')

        if self.app_limited:
            self.cwnd = (self.cwnd + self.inflight) / 2.0  # RFC2861
        else:
            self.prev_cwnd = self.cwnd
            self.prev_action = self.action
            r_action = random.randint(0,4)
            action = self.sample_action(h_state) 
            #action = 4 if not self.slow_start and action == 3 else action
            if self.rand_rate ==0 or random.random() > self.rand_rate:
                self.__take_action(action)
            else:
                self.__take_action(r_action)
            self.action = action
        #print self.cwnd
        #print('bytes_sent = {},bytes_acked ={}'.format(self.bytes_sent,self.bytes_acked ))
        # reset at the end of each step

        #up_load ststistics 
        if self.perf_client:
            self.perf_client.collect_perf_data(self)

        self.__reset_step()

        # step counting
        if self.train:
            self.step_num += 1
            if self.step_num >= Policy.steps_per_episode:
                self.__episode_ended()

    def __is_step_ended(self, duration):
        if self.train:
            return duration >= Policy.min_step_len
        else:
            # cwnd is updated every RTT in start phase, and min_rtt/action_frequency afterwards
            if self.start_phase_cnt < self.start_phase_max:
                threshold = max(self.min_rtt, Policy.min_step_len)
                self.start_phase_cnt += 1
            else:
                threshold = max(self.min_rtt / Policy.action_frequency, Policy.min_step_len)

            return duration >= threshold

    def __re_update_rtt(self, curr_ts):
        if self.rtt_reupdate_start_ts is None:
            self.rtt_reupdate_start_ts = curr_ts
            return

        if curr_ts - self.rtt_reupdate_start_ts > self.rtt_reupdate_interval * self.min_rtt:
            print curr_ts - self.rtt_reupdate_start_ts, 'Min RTT:', self.min_rtt, 'Max RTT:', self.max_rtt
            self.rtt_reupdate_start_ts = curr_ts
            self.min_rtt = sys.maxint
            self.max_rtt = 0

# public:
    def ack_received(self, ack):
        self.ack_recv_ts = timestamp_ms()
        if Policy.delay_ack:
            self.bytes_acked += Message.total_size * Policy.delay_ack_count
        else:
            self.bytes_acked += Message.total_size

        self.__update_state(ack)

        curr_ts = timestamp_ms()
        #cal_before = timestamp_us()
        if self.step_start_ts is None:
            self.step_start_ts = curr_ts
        # check if the current step has ended
        if self.__is_step_ended(curr_ts - self.step_start_ts):
            #sys.stderr.write("step ended\n")

            self.step_start_ts = curr_ts

            self.__step_ended()
            #prev_m_timestamp = self.m_timestamp
            #self.m_timestamp = timestamp_us()
            #print('model calling step interval {}us'.format(self.m_timestamp - prev_m_timestamp))
            #cal_after = curr_ts_us = timestamp_us()
            #print('step call time = {}us'.format(cal_after- cal_before))

        # self.__re_update_rtt(curr_ts)

    def data_sent(self):
        self.bytes_sent += Message.total_size

    def timeout_ms(self):
        return 100

    def set_sample_action(self, sample_action):
        self.sample_action = sample_action

    def set_perf_client(self, perf_client):
        self.perf_client = perf_client

    def pacing_pkt_number(self, max_in_cwnd):
        # pacing control
        max_in_cwnd = int(max_in_cwnd)

        if not self.pacing or self.min_rtt == sys.maxint:
            return max_in_cwnd

        if self.pre_sent_ts is None:
            self.pre_sent_ts = datetime.datetime.now()

        now = datetime.datetime.now()
        duration = (now - self.pre_sent_ts).microseconds
        if duration >= self.interval:
            pacing_rate = self.cwnd / self.min_rtt  # pkt/ms
            n = duration / 1000.0 * pacing_rate

            self.borrowed_pkt += n - int(n)
            n = int(n)
            if self.borrowed_pkt >= 1.0:
                n += int(self.borrowed_pkt)
                self.borrowed_pkt = self.borrowed_pkt - int(self.borrowed_pkt)
            ret_num = min(max_in_cwnd, n)

            self.pre_sent_ts = now
        else:
            ret_num = 0

        return ret_num
    def set_env_name(self,env_name):
        self.env_name = env_name
        print('policy config  env {}'.format(env_name))
        (self.env_bandwidth,self.env_delay) = map(lambda x:float(x),env_name.replace('env_','').replace('ms','').split('_'))
        self.max_bdp_norm = (self.env_bandwidth*self.env_delay*2)/(Policy.max_send_rate*Policy.max_rtt)
        print('policy config  bandwidth {},rtt {},bdp_norm {}'.format(self.env_bandwidth,self.env_delay,self.max_bdp_norm))
    
    def set_sender(self,sender):
        self.sender = sender