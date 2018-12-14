import glob
import cv2
import numpy as np 
import tensorflow as tf
import time
import os


checkpoint_dir = './result_dir/'
result_dir = './result_dir/'
input_dir = './data/Set12/'
patch = 40
sigma = 25

def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr
def dncnn(input, is_training=True, output_channels=1):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 16 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block17'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return output

sess = tf.Session()
clean_image = tf.placeholder(tf.float32,[None,None,None,1],name = 'clean_image')
noise = tf.random_normal(shape = tf.shape(clean_image),stddev = sigma/255.0)
in_image = clean_image + noise
is_training = tf.placeholder(tf.bool,name = 'is_training')
Res = dncnn(in_image,is_training)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

if ckpt:
	print('loaded' + ckpt.model_checkpoint_path)
	saver.restore(sess,ckpt.model_checkpoint_path)

walks = glob.glob(input_dir+'*.png')
test_id = 0
if not os.path.isdir(result_dir+'test'):
	os.makedirs(result_dir+'test')
for walk in walks:
	print(walk)
	test_img = cv2.imread(walk,0)
	# test_img = cv2.resize(test_img,(patch,patch),interpolation = cv2.INTER_CUBIC)
	test_img = test_img.astype(np.float32)/255.0
	test_img = np.expand_dims(test_img,axis = 0)
	test_img = np.expand_dims(test_img,axis = 3)
	res,noise_img = sess.run([Res,in_image],feed_dict={clean_image:test_img,is_training:False})
	res = noise_img - res
	print(cal_psnr(res[0,:,:,:]*255,test_img[0,:,:,:]*255))
	print(cal_psnr(noise_img[0,:,:,:]*255,test_img[0,:,:,:]*255))
	cv2.imwrite(result_dir+'test/%05d_res.png' %(test_id),res[0,:,:,:]*255)
	cv2.imwrite(result_dir+'test/%05d_noise_img.png' %(test_id),noise_img[0,:,:,:]*255)
	cv2.imwrite(result_dir+'test/%05d.png' %(test_id),test_img[0,:,:,:]*255)
	test_id += 1




