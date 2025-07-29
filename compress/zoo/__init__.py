# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from compress.models import SymmetricalTransFormer, WACNN,  WACNN_CLRSUP_T, WACNN_CLRSUP_D, WACNN_CLRSUP_F, WACNN_CLRSUP_F_G

from .pretrained import load_pretrained as load_state_dict

models = {
    'stf': SymmetricalTransFormer,
    'cnn': WACNN,
    'super_pixel_color_t': WACNN_CLRSUP_T,
    # super_pixel采样 简单着色网络
    'super_pixel_color_d': WACNN_CLRSUP_D,
    # super_pixel采样 可变形着色网络
    'super_pixel_color_f': WACNN_CLRSUP_F,
    # super_pixel采样 可变形着色网络+融合网络
    'super_pixel_color_f_g': WACNN_CLRSUP_F_G,
    # super_pixel采样 可变形着色网络+融合网络
}
