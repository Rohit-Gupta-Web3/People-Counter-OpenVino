"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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


import os
import sys
import time
import socket
import json
import math
import cv2
import numpy as np
import logging as log
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="Path to an xml file with a trained model.",
    )
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="Path to image or video file"
    )
    parser.add_argument(
        "-l",
        "--cpu_extension",
        required=False,
        type=str,
        default=CPU_EXTENSION,
        help="MKLDNN (CPU)-targeted custom layers."
        "Absolute path to a shared library with the"
        "kernels impl.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="CPU",
        help="Specify the target device to infer on: "
        "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
        "will look for a suitable plugin for device "
        "specified (CPU by default)",
    )
    parser.add_argument(
        "-pt",
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for detections filtering" "(0.5 by default)",
    )
    return parser


def connect_mqtt():
    """
                This method intends to create and deploy a MQTT client
        :return: MQTT client
    """
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def draw_outputs(coords, frame, initial_w, initial_h, x, k, prob_threshold):
    """
                This method is intended to return the frame with bounding boxes and other details
                
        :param: Inference Output
        :param: current frame
        :param: width of the frame
        :param: height of the frame
        :param: minimum threshold value
        :return: processed frame,
        :return: current count
        :return: lenght of arrow
        :return: previous count
    """
    current_count = 0
    ed = x
    for obj in coords[0][0]:
        # Draw bounding box for object when it's probability is more than the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            current_count = current_count + 1

            x = int((xmin + xmax) / 2)
            y = int((ymax + ymin) / 2)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            x = int((xmin + xmax) / 2)
            y = int((ymax + ymin) / 2)
            if x != 0 and y != 0:
                cv2.putText(
                    frame,
                    "Person Detected",
                    (100, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (100, 100, 255),
                    1,
                )
                cv2.arrowedLine(frame, (160, 50), (x, y), (255, 0, 255), 3)
                c_x = frame.shape[1] / 2
                c_y = frame.shape[0] / 2
                mid_x = (160 + x) / 2
                mid_y = (50 + y) / 2

                # Calculating distance
                ed = math.sqrt(
                    math.pow(mid_x - c_x, 2) + math.pow(mid_y - c_y, 2) * 1.0
                )
            k = 0

    if current_count < 1:
        k += 1

    if ed < 110 and k < 10:
        current_count = 1
        k += 1
        if k > 100:
            k = 0

    return frame, current_count, ed, k


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network(
        model=args.model, device=args.device, cpu_ex=args.cpu_extension
    )
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    temp = 0
    tk = 0
    last_count = 0
    total_count = 0

    net = infer_network.load_model()
    net_input_shape = infer_network.get_input_shape()
    net_output_shape = infer_network.get_output_shape()

    image_flag = False
    if args.input == "CAM":
        args.input = 0
    elif args.input.endswith(".jpg") or args.input.endswith(".bmp"):
        image_flag = True

    stream = cv2.VideoCapture(args.input)
    stream.open(args.input)

    # shape of the input loaded by opencv
    initial_w = int(stream.get(3))
    initial_h = int(stream.get(4))

    while stream.isOpened():

        flag, frame = stream.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, 3, net_input_shape[2], net_input_shape[3])

        infer_network.exec_net(p_frame)
        count = 0

        # Start asynchronous inference for specified request
        inf_start = time.time()
        infer_network.exec_net(p_frame)

        color = (255, 0, 0)
        perf = infer_network.layer_wise()
        # os.chdir("../reports/")
        if not len(perf) == 0:
            f = open("model.json", "w")
            f.write(json.dumps(perf, indent=4))
            f.close()

        # Wait for the result
        if infer_network.wait() == 0:
            det_time = time.time() - inf_start

            # Get the results of the inference request
            result = infer_network.get_output()

            # Draw Bounting Box
            frame, current_count, d, tk = draw_outputs(
                result, frame, initial_w, initial_h, temp, tk, prob_threshold
            )

            # Printing Inference Time
            inf_time_message = "Inference time: {:.3f}ms".format(det_time * 1000)
            cv2.putText(
                frame,
                inf_time_message,
                (15, 15),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                color,
                1,
            )

            # Calculate and send relevant information
            if current_count > last_count:  # New entry
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))

            if current_count < last_count:  # Average Time
                duration = int(time.time() - start_time)
                client.publish("person/duration", json.dumps({"duration": duration}))

            if current_count > 1:
                client.publish("person", json.dumps({"count": 1}))
            else:
                client.publish(
                    "person", json.dumps({"count": current_count})
                )  # People Count

            last_count = current_count
            temp = d

        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        if key_pressed == 27:
            break

        if image_flag:
            cv2.imwrite("output_image.jpg", p_frame)

    stream.release()
    cv2.destroyAllWindows()
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == "__main__":
    main()
