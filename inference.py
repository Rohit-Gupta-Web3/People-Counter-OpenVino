#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os  # used to split the model location string
import time  # used to provide delay when waiting for the async request to finish
from openvino.inference_engine import (
    IENetwork,
    IEPlugin,
)  # used to load the IE python API
import logging as log


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self, model, device="CPU", cpu_ex=None):
        """
                This method intends to initialize all the attributes of the class
        :param model: The name of model .xml file
        :param device: Device Type(CPU/GPU/VPU/GPGA)
        :param cpu_ex: CPU extension path
        """
        self.ie = None
        self.net = None
        self.inp = None
        self.out = None
        self.ext = None
        self.ex_net = None
        self.supported = None
        self.device = device
        self.cpu_ex = cpu_ex
        self.model = model

    def load_model(self):
        """
                This method intends to load the model
        :return: Network
        """
        self.ie = IEPlugin(self.device)
        model_xml = self.model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.net = IENetwork(model=model_xml, weights=model_bin)
        
        self.check_cpu_support()
        
        if self.cpu_ex and "CPU" in self.device:
            self.ie.add_cpu_extension(self.cpu_ex)
        log.info("After loading CPU extension\n")
        self.check_cpu_support()
        self.ex_net = self.ie.load(self.net)
        
        # to get the shape of input and output and set each to a class variable
        self.inp = next(iter(self.net.inputs))
        self.out = next(iter(self.net.outputs))

        # Note: You may need to update the function parameters. ###
        return self.ex_net
    
    def check_cpu_support(self):
        self.supported = self.ie.get_supported_layers(self.net)
        unsupported_layers = [
            layer for layer in self.net.layers.keys() if layer not in self.supported
        ]
        log.error("list of unsupported layers: {}".format(unsupported_layers))

    def get_input_shape(self):
        """
                This method intends to set the input shape parameter
        :return: input shape parameter
        """
        return self.net.inputs[self.inp].shape

    def exec_net(self, image):
        """
                This method intends execute the inference network
        :image: current frame
        :return: executed network
        """
        self.ex_net.start_async(request_id=0, inputs={self.inp: image})
        while True:
            status = self.wait()
            if status == 0:
                break
            else:
                time.sleep(-1)
        return self.ex_net

    def wait(self):
        """
                This method intends to set the app into the blocking state untill the inference is completed
        :return: blocking state
        """
        return self.ex_net.requests[0].wait(-1)

    def get_output(self):
        """
                This method intends to return the processed output
        :return: processed output
        """
        return self.ex_net.requests[0].outputs[self.out]

    def get_output_shape(self):
        """
                This method intends to set the output shape parameter
        :return: output shape parameter
        """
        self.output_shape = self.net.outputs[self.out].shape

    def layer_wise(self):
        """
                This method returns the per layer CPU time
        :return: layer wise CPU time
        """
        return self.ex_net.requests[0].get_perf_counts()
