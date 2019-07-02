#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:33:23 2019

@author: sleek_eagle
"""

import alexnet
import numpy as np

model = alexnet.alexnet_model_TP()
model.summary()

ar = np.random.rand(100,224,224,1)
model.predict(ar)
