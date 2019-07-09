#!/usr/bin/env python
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

from helpers.utils import timestamp_ms


class App(object):
    max_sendbuf_size = 8192

    def __init__(self, app_type):
        self.app_type = app_type

        # app's send buffer
        self.send_buffer = 0

    def start(self):
        self.start_ts = timestamp_ms()
        self.pre_update_ts = self.start_ts

    def flush_app(self):
        now = timestamp_ms()

        # rate of application generating packet, according to app_type
        # an example, fixed rate
        self.app_flow_rate = 10  # pkt/ms
        self.send_buffer += (now - self.pre_update_ts) * self.app_flow_rate

        self.send_buffer = min(App.max_sendbuf_size, self.send_buffer)

        self.pre_update_ts = now

    def get_app_data(self):

        if self.send_buffer > 0:
            self.send_buffer -= 1
            return True
        else:
            return False
