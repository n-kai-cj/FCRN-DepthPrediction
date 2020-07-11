import argparse
import time

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='pretrained/NYU_FCRN.ckpt', help="pre-trained model")
parser.add_argument('-i', '--id', type=int, default=0, help="camera id")
parser.add_argument('-r', '--resolution', type=str, default='640x360', help="capture resolution")
args = parser.parse_args()

model = args.model
cam_id = args.id
try:
    width = int(args.resolution.split('x')[0])
    height = int(args.resolution.split('x')[1])
except:
    width = 640
    height = 480

# Tensorflow Input Size
inW = 304
inH = 228
channels = 3
batch_size = 1

tfload_start = time.time()
# Create a placeholder for the input image
input_node = tf.placeholder(tf.float32, shape=(None, inH, inW, channels))
tfload_0 = time.time()

# Construct the network
net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
tfload_1 = time.time()

# Load the converted parameters
print('Loading the model')

sess = tf.Session()
tfload_2 = time.time()
# Use to load from ckpt file
saver = tf.train.Saver()
saver.restore(sess, model)
tfload_3 = time.time()
print("Tf Load {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
    tfload_0 - tfload_start, tfload_1 - tfload_start, tfload_2 - tfload_start, tfload_3 - tfload_start))

# Prepare Video Capture
cap = cv2.VideoCapture(cam_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, src_frame = cap.read()
    if not ret:
        break

    frame = src_frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (inW, inH), interpolation=cv2.INTER_AREA)
    img = np.array(frame).astype('float32')
    img = np.expand_dims(np.asarray(img), axis=0)

    # Evalute the network for the given image
    pred_start = time.time()
    pred = sess.run(net.get_output(), feed_dict={input_node: img})
    pred_end = time.time()
    print("predict time = {:.3f}[sec]".format(pred_end - pred_start))

    # Plot result
    depth = pred[0, :, :, 0]
    color_depth = utils.colored_depthmap(depth).astype('uint8')
    color_depth = cv2.resize(color_depth, (width, height))
    out = cv2.hconcat([src_frame, color_depth])
    cv2.imshow("", out)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("--- finish ---")
