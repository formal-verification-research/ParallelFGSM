#!/usr/bin/env python3

from make_mnist_cnn_tf import build_cnn_mnist_model, reset_graph
import tensorflow as tf
import numpy as np
import time
import argparse
import re
from random import seed
from random import randint

file = open("random.txt", "w+")
seed(10)
for test_value in range(10):
    curr_value = randint(1, 60000) 
    # if curr_value == 0:
        # initial_value = 1
        # initial_value = 0
    # elif curr_value == 1:
        # initial_value = 3
    # elif curr_value ==  2:
        # initial_value = 5 
    # elif curr_value == 3:
        # initial_value = 7
    # elif curr_value == 4:
        # initial_value = 2
    # elif curr_value == 5:
        # initial_value = 0
    # elif curr_value == 6:
        # initial_value = 13
    # elif curr_value == 7:
        # initial_value = 15
    # elif curr_value == 8:
        # initial_value = 17
    # elif curr_value == 9:
        # initial_value = 4
    
    for curr_target in range(10):
        if curr_target == 0:
            target = 1
        elif curr_target == 1:
            target = 3
        elif curr_target ==  2:
            target = 5 
        elif curr_target == 3:
            target = 7
        elif curr_target == 4:
            target = 2
        elif curr_target == 5:
            target = 0
        elif curr_target == 6:
            target = 13
        elif curr_target == 7:
            target = 15
        elif curr_target == 8:
            target = 17
        elif curr_target == 9:
            target = 4
            
        count_fails = 0
        count_target = 0
        count_zero = 0
        count_one =0
        count_two = 0
        count_three = 0
        count_four = 0
        count_five = 0
        count_six = 0
        count_seven = 0
        count_eight = 0
        count_nine =0

        #0 1 2 3 4 5 6  7  8  9
        #1 3 5 7 2 0 13 15 17 4
        parser = argparse.ArgumentParser()
        parser.add_argument('--epsmin', type=float, default=0.01)
        parser.add_argument('--epsmax', type=float, default=0.2)
        parser.add_argument('--idx', type=int, default=100)
        parser.add_argument('--numgens', type=int, default=1000)

        args = parser.parse_args()

        reset_graph()
        x = tf.placeholder(tf.float32, shape=(None, 28, 28))
        y = tf.placeholder(tf.int32, shape=(None,))
        model = build_cnn_mnist_model(x, y, False)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train / np.float32(255)
        y_train = y_train.astype(np.int32)
        x_test = x_test / np.float32(255)
        y_test = y_test.astype(np.int32)

        grad, = tf.gradients(model['loss'], x)
        epsilon = tf.placeholder(tf.float32)
        optimal_perturbation = tf.multiply(-1*tf.sign(grad), epsilon)#change
        adv_example_unclipped = tf.add(optimal_perturbation, x)
        adv_example = tf.clip_by_value(adv_example_unclipped, 0.0, 1.0)

        classes = tf.argmax(model['probability'], axis=1)

        adv_examples = []
        idx = curr_value#args.idx#change
       
        epsilon_range = (args.epsmin, args.epsmax)
        #for i in range(100):
        #    print(y_train[i])
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        with tf.Session(config = config) as sess:
            saver.restore(sess, './models/mnist_cnn_tf/mnist_cnn_tf')
            acc_test = model['accuracy'].eval(feed_dict={
                x: x_test,
                y: y_test,
            })
            print('Accuracy of model on test data: {}'.format(acc_test))
            print('Correct Class: {}'.format(y_train[idx]))
            class_x = classes.eval(feed_dict={x: x_train[idx:idx + 1]})
            print('Predicted class of input {}: {}'.format(idx, class_x))
            start = time.time()
            for i in range(args.numgens):
                adv = adv_example.eval(
                    feed_dict={
                        x: x_train[idx:idx + 1],
                        y: y_train[target:target  + 1], #y_train[idx:idx + 1],#change
                        epsilon: np.random.uniform(
                            epsilon_range[0], epsilon_range[1],
                            # size=(28, 28)
                            )
                    })
                class_adv = classes.eval(feed_dict={x: adv, y: y_train[idx:idx+1]})
                if class_adv != y_train[idx]:#detirmines true class
                    adv_examples += [adv]
                    if class_adv == 0:
                        count_zero = count_zero +1
                    if class_adv == 1:
                        count_one = count_one +1
                    if class_adv == 2:
                        count_two = count_two +1
                    if class_adv == 3:
                        count_three = count_three +1
                    if class_adv == 4:
                        count_four = count_four +1
                    if class_adv == 5:
                        count_five = count_five +1
                    if class_adv == 6:
                        count_six = count_six +1
                    if class_adv == 7:
                        count_seven = count_seven +1
                    if class_adv == 8:
                        count_eight = count_eight +1
                    if class_adv == 9:
                        count_nine = count_nine +1

                    if class_adv == y_train[target]:
                        count_target = count_target+1
                else:
                    count_fails = count_fails +1
            print('Duration (s): {}'.format(time.time() - start))
            print("source ", curr_value)
            print("curr_target", curr_target)
        if adv_examples != []:
            adv_examples = np.concatenate(adv_examples, axis=0)
            print('Found {} adversarial examples.'.format(adv_examples.shape[0]))
            print('Percentage true adversarial examples: {}'.format(adv_examples.shape[0]/args.numgens))
            avg = np.zeros_like(x_train[idx])
            for i in range(adv_examples.shape[0]):
                avg += adv_examples[i]
            avg /= adv_examples.shape[0]
            stddev = 0
            for i in range(adv_examples.shape[0]):
                stddev += np.linalg.norm((adv_examples[i] - avg).flatten())

            stddev /= adv_examples.shape[0]
            print('Found std dev: {}'.format(stddev))
            print("failed ", count_fails)
            file.write('Y_train[' +str(idx) + ']\n')
            file.write('source ' +str(y_train[idx]) + '\n')
            file.write('target ' +str(curr_target) + '\n')
            file.write('success ' +str(count_target) + '\n')
            file.write('generated ' +str(count_zero) + ' zeros\n')
            file.write('generated ' + str(count_one) +' ones\n')
            file.write('generated '+ str(count_two) +' twos\n')
            file.write('generated '+ str(count_three) +' threes\n')
            file.write('generated '+ str(count_four)+' fours\n') 
            file.write('generated '+ str(count_five)+' fives\n')
            file.write('generated '+ str(count_six)+' sixes\n')
            file.write('generated '+ str(count_seven)+' sevens\n')
            file.write('generated '+ str(count_eight)+' eights\n')
            file.write('generated '+ str(count_nine)+' nines')
            
            print("count zero", count_zero)
            print("count one ", count_one)
            print("count two ", count_two)
            print("count three ", count_three)
            print("count four ", count_four)
            print("count five ", count_five)
            print("count six ", count_six)
            print("count seven ", count_seven)
            print("count eight ", count_eight)
            print("count nine ", count_nine)
            print("hit target ", count_target)
        print(test_value, "\n")
        file.write('\n\n\n')    
file.close()
            