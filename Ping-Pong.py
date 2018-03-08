import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import math
import random
import glob
import tensorflow as tf
import os
#background is rwo,column
#v is vertical,horizontal

# Write the correct physical laws

fig = plt.figure()
ax = fig.add_subplot(111)

radius =10

LR = 0.0005
Step_size = 10
initial_velocity = [10,10]
length = 534
breadth = 256
bat_size = 40
max_vel = 10

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def get_background():
	RGB = [255,255,255]
	RGBlen = []
	for i in range(length):
		RGBlen.append(RGB)
	white = []
	for i in range(breadth):
		white.append(RGBlen)
	white = np.array(white).astype(np.uint8)
	for i in range(breadth):
		white[i][5:10] = [0,0,0]
	return white

def plot(r_bat,r_ball,up_down,v_ball,background,im,backup):
	save1 = []
	for i in range(2*radius):
		for j in range(2*radius):
			if (radius -1*i)*(radius -1*i)+(radius -1*j)*(radius -1*j)<radius*radius:
				save1.append([int(r_ball[0])-1*radius+i,int(r_ball[1])-1*radius+j])
				background[int(r_ball[0])-1*radius+i][int(r_ball[1])-1*radius+j]=[255,0,0]
	for i in range(bat_size):
		background[int(r_bat)-1*bat_size//2+i][length -15:length -10]=[255,0,0]
	im.set_data(background)
	fig.canvas.draw()
	for i in range(len(save1)):
		background[save1[i][0]][save1[i][1]]=backup[save1[i][0]][save1[i][1]]
	for i in range(bat_size):
		background[int(r_bat)-1*bat_size//2+i][length -15:length -10]=backup[int(r_bat)-1*bat_size//2+i][length -15:length -10]

def change_bat(v_ball):
	v_ball[1] = -1*v_ball[1]
	return v_ball
def change_up_down(v_ball):
	v_ball[0] = -1*v_ball[0]
	return v_ball
def change_wall(v_ball):
	# v_ball[1] = -1*v_ball[1]
	v_ball[1] = random.randint(max_vel-1,max_vel)
	#print(v_ball[1])
	return v_ball

def move(r_bat,r_ball,up_down,v_ball):
	val = 0
	r_bat = r_bat + up_down
	pos = 0
	if r_bat + bat_size/2 >= breadth:
		r_bat = r_bat -1*up_down
	elif r_bat -1*bat_size/2<0:
		r_bat = r_bat -1*up_down
	r_ball_new = r_ball + v_ball
	if r_ball_new[1] -1*radius < 10:
		v_ball = change_wall(v_ball)
	elif r_ball_new[0] -1*radius<0:
		v_ball = change_up_down(v_ball)
	elif r_ball_new[0] + radius>=breadth:
		v_ball = change_up_down(v_ball)
	elif r_ball_new[1] + radius >= length -1*20 and r_ball_new[0] + radius>=r_bat -1*bat_size/2 and r_ball_new[0] -1*radius<=r_bat + bat_size/2:
		val = 1
		v_ball = change_bat(v_ball)
		r_ball = r_ball + v_ball
		while(True):
			if r_ball[1] + radius >= length -1*20 and r_ball[0] + radius>=r_bat -1*bat_size/2 and r_ball[0] -1*radius<=r_bat + bat_size/2:
				r_ball = r_ball + v_ball
			else:
				break
		pos = r_ball[0]

	elif r_ball_new[1] + radius >= length:
		val = -1
		pos = r_ball_new[0]
	else:
		r_ball = r_ball_new
	return val,r_bat,r_ball,v_ball,pos

def make_answer(r_bat,r_ball,v_ball):
	answer = np.array([r_bat] + [r_ball[0]] + [r_ball[1]] + [v_ball[0]] + [v_ball[1]])
	return answer

def change_v_choice(decision,v):
	val = np.max(decision)
	actually_taken = 0
	up_down = 0
	if val == decision[0]:
		actually_taken = 0
		up_down = Step_size
	elif val == decision[1]:
		actually_taken = 1
		up_down = -Step_size
	else:
		actually_taken = 2
		up_down = 0
	return up_down,actually_taken

def make_batch(Steps,pos,number_of_games):
	Batch = []
	temp0 = []
	temp1 = []
	for i in range(len(Steps)):
		temp0.append(Steps[i][0])
		y = Steps[i][2][0]
		t = [0,0,0]
		if Steps[i][0][0] < pos +bat_size//2 and Steps[i][0][0]>pos - bat_size//2:
			t[2] = 1
		elif Steps[i][0][0]>pos:
			t[1] = 1
		elif Steps[i][0][0]<pos:
			t[0] = 1
		temp1.append(t)
		#print(Steps[i][1])
		# print(y,'chocie taken')
		# print(pos,'ball pos in end')
		# print(Steps[i][0][0],'pos of bat in the game')
		#print(t,'decision taken')
		# os.system('pause')

	Batch.append(temp0)
	Batch.append(temp1)
	return Batch

def tensor():
	r_ball = np.array([128,256])
	r_bat = 128
	v_ball = initial_velocity
	up_down = 0

	inp_1 = tf.placeholder(dtype=tf.float32, shape=[None,5])
	out_1 = tf.placeholder(dtype=tf.float32, shape=[None,3])

	W_1_1 = weight_variable([5, 2048])
	b_1_1 = bias_variable([2048])

	# W_2_1 = weight_variable([1024,512])
	# b_2_1 = bias_variable([512])

	W_3_1 = weight_variable([2048,3])
	b_3_1 = bias_variable([3])

	first_layer_1 = tf.nn.sigmoid(tf.matmul(inp_1, W_1_1) + b_1_1)
	#second_layer_1 = tf.nn.sigmoid(tf.matmul(first_layer_1, W_2_1) + b_2_1)
	third_layer_1 = tf.nn.softmax(tf.matmul(first_layer_1, W_3_1) + b_3_1)
	loss = -tf.reduce_sum(tf.multiply(tf.log(third_layer_1),out_1))
	optmizer = tf.train.AdamOptimizer(LR).minimize(loss)

	sess = tf.InteractiveSession()
	init = tf.global_variables_initializer().run()

	white = get_background()
	backup = np.copy(white)

	im = plt.imshow(white)

	Steps = []
	Big_Batch = []
	Good_Batch = []
	number_of_games = 0
	hit = 0
	first_time = 1
	shower = 0
	while(True):
		if number_of_games % 200 is 0 or shower == 1:
			plot(r_bat,r_ball,up_down,v_ball,white,im,backup)
		pos = 0
		temp1 = make_answer(r_bat,r_ball,v_ball)
		temp4 = []
		temp4.append(temp1)
		decision = third_layer_1.eval(feed_dict={inp_1: temp4})
		temp2 = np.array([decision[0]])

		up_down,actually_taken = change_v_choice(decision[0],up_down)

		temp5 = [actually_taken]

		temp3 = [temp1,temp2,temp5]

		Steps.append(temp3)

		val,r_bat,r_ball,v_ball,pos = move(r_bat,r_ball,up_down,v_ball)
		if val is not 0:
			if hit > 0 and hit%1000 == 0:
				print(hit)
				print(number_of_games)
				shower = 1
			if val is 1:
				hit = hit + 1
			Batch = make_batch(Steps,pos,number_of_games)
			if val is -1:
				if number_of_games%200 is 0:
					print('number_of_games',number_of_games)
					print('hit',hit)
					hit = 0
				number_of_games = number_of_games + 1
			if first_time is 1:
				Big_Batch = Big_Batch + Batch
				first_time = 0
			else:
				Big_Batch[0] = Big_Batch[0] + Batch[0]
				Big_Batch[1] = Big_Batch[1] + Batch[1]
			if val is -1:
				r_ball = np.array([random.randint(15 + radius,128),random.randint(15 + radius,256)])
				v_ball = initial_velocity
				r_bat = 128

				while(True):
					x1 = random.randint(-1,1)
					x2 = random.randint(-1,1)
					if x1 is not 0 and x2 is not 0:
						v_ball[0] = v_ball[0]*x1
						v_ball[1] = v_ball[1]*x2
						break
						
				if number_of_games%5 is 0 and number_of_games is not 0:
					optmizer.run(feed_dict={inp_1: Big_Batch[0],out_1: Big_Batch[1]})
					print('trained')
					Big_Batch[:] = []
					first_time = 1


			Steps[:] = []

win = fig.canvas.manager.window
fig.canvas.manager.window.after(10, tensor)
plt.show()	