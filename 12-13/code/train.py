import glob
import cv2
import numpy as np 
import tensorflow as tf
import time
import os


input_dir = './data/train/'
checkpoint_dir = './result_dir/'
result_dir = './result_dir/'
test_dir = './data/Set12/'
log_dir = './result_dir/'
initializer = tf.truncated_normal_initializer(stddev=0.02)

patch_size, stride = 40, 10
#scales = [1, 0.9, 0.8, 0.7]
scales = [0.7]
aug_times = 1
batch_size = 128
sigma = 25
epochs = 30
learning_rate = 1e-3 * np.ones([epochs])
learning_rate[6:] = learning_rate[0] / 10.0
learning_rate[20:] = learning_rate[7] / 10.0
save_freq = 5

def gen_patches(file_name):
	img = cv2.imread(file_name,0)
	h, w = img.shape
	patches = []
	for s in scales:
		h_scaled, w_scaled = int(h*s), int(w*s)
		img_scaled = cv2.resize(img,(h_scaled,w_scaled),interpolation = cv2.INTER_CUBIC)
		for i in range(0,h_scaled - patch_size + 1, stride):
			for j in range(0,w_scaled - patch_size + 1, stride ):
				x = img_scaled[i:i+patch_size,j:j+patch_size]
				for k in range(0,aug_times):
					mode = np.random.randint(0,8)
					if mode == 0 :
						x_aug = x
					elif mode == 1:
						x_aug = np.flipud(x)
					elif mode == 2:
						x_aug = np.rot90(x) 
					elif mode == 3:
						x_aug = np.flipud(np.rot90(x))
					elif mode == 4:
						x_aug = np.rot90(x,k =2)
					elif mode == 5:
						x_aug = np.flipud(np.rot90(x,k=2))
					elif mode == 6:
						x_aug = np.rot90(x,k=3)
					elif mode == 7:
						x_aug = np.flipud(np.rot90(x,k=3))
					patches.append(x_aug)
	return patches

def datagenerator(data_dir):
	file_list = glob.glob(data_dir+'*.png')
	print(len(file_list))
	data = []
	for i in file_list:
		patches = gen_patches(i)
		for patch in patches:
			data.append(patch)
	clean = []
	for d in data:
		if d.shape not in clean:
			clean.append(d.shape)
	data = np.array(data,dtype='uint8')
	data = np.expand_dims(data,axis = 3)
	discard_n = len(data) - len(data) // batch_size *batch_size
	data = np.delete(data,range(discard_n),axis = 0)
	print(data.shape)
	if(data.shape[0]>60000):
		return []
	print("training data finished!")
	return data

data = datagenerator(data_dir = input_dir)
if(len(data)==0):
	print('data too large')
data = data.astype(np.float32)/255.0

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

def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))

sess = tf.Session()
clean_image = tf.placeholder(tf.float32,[None,None,None,1],name = 'clean_image')
noise = tf.random_normal(shape = tf.shape(clean_image),stddev = sigma/255.0)
in_image = clean_image + noise
is_training = tf.placeholder(tf.bool,name = 'is_training')
Res = dncnn(in_image,is_training)

G_loss = (1.0/batch_size)*tf.nn.l2_loss(noise- Res)
lr = tf.placeholder(tf.float32,name = 'learning_rate')
G_opt = tf.train.AdamOptimizer(lr,name = 'AdamOptimizer').minimize(G_loss)
Psnr = tf_psnr(clean_image, in_image - Res)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
numBatch = int(data.shape[0]/batch_size)
if ckpt:
	print('loaded' + ckpt.model_checkpoint_path)
	saver.restore(sess,ckpt.model_checkpoint_path)
f = open(log_dir+'log','w+')
for epoch in range(0,epochs):
	np.random.shuffle(data)
	for batch_id in range(0,numBatch):
		batch_images = data[batch_id * batch_size : (batch_id + 1) * batch_size ,:,:,:]
		start_time = time.time()
		_,loss =  sess.run([G_opt,G_loss],feed_dict = {clean_image : batch_images,lr : learning_rate[epoch],is_training : True})
		print_vec = "Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
			% (epoch, batch_id, numBatch, time.time() - start_time, loss)
		print(print_vec)
		print >> f,print_vec
	if epoch % save_freq == 0:
		if not os.path.isdir(result_dir + '%04d' %epoch):
			os.makedirs(result_dir + '%04d' % epoch)
		test_file_list = glob.glob(test_dir + '*.png')
		train_id = 0
		for i in test_file_list:
			test_img = cv2.imread(i,0).astype(np.float32)/255.0
			test_img = np.expand_dims(test_img,axis = 0)
			test_img = np.expand_dims(test_img,axis = 3)
			r,n ,p=  sess.run([Res,in_image,Psnr],feed_dict={clean_image: test_img,lr:learning_rate[epoch],is_training:False})
			r = n-r
			print_vec = "Epoch: [%2d] PSNR: %.6f"%(epoch+1,p)
			print(print_vec)
			print >> f,print_vec
			temp = np.concatenate((test_img[0,:,:,:],r[0,:,:,:]),axis = 1)
			cv2.imwrite(result_dir + '%04d/%05d.jpg' % (epoch, train_id), temp*255)
			train_id += 1
saver.save(sess,checkpoint_dir+'model.ckpt')