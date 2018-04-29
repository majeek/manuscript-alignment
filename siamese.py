import numpy as np
import codecs
import pickle
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Lambda, Flatten, Input, MaxPooling2D, Convolution2D
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from scipy import misc
import os
from os import listdir
from os.path import isfile, join
import random
import itertools


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def shuffle_and_limit(pairs, limit):
    filter_dec = {}
    for word_part in pairs.keys():
        good_pairs = pairs[word_part]
        np.random.shuffle(good_pairs)
        filter_dec[word_part] = good_pairs[:limit]

    return filter_dec


def create_true_pairs(word_part_imgs_b1, word_part_imgs_b2, min_instances):
    true_pairs = {}
    for word_part in word_part_imgs_b1.keys():
        b1_instances = list(word_part_imgs_b1[word_part])
        b2_instances = list(word_part_imgs_b2[word_part])
        if len(b1_instances) * len(b2_instances) >= min_instances:
            for b1_instance in b1_instances:
                for b2_instance in b2_instances:
                    if word_part in true_pairs.keys():
                        true_pairs[word_part].append([b1_instance, b2_instance])
                    else:
                        true_pairs[word_part] = [[b1_instance, b2_instance]]
    return true_pairs


def create_false_pairs(word_part_imgs_b1, word_part_imgs_b2, true_pairs):
    false_pairs = {}
    for word_part in true_pairs.keys():
        positive_instances = len(true_pairs[word_part])
        b1_instances = list(word_part_imgs_b1[word_part])
        b2_instances = []
        for b2_word_part in word_part_imgs_b2:
            if b2_word_part != word_part:
                b2_instances.extend(word_part_imgs_b2[b2_word_part])
        b2_instances = list(set(b2_instances))
        false_pairs[word_part] = []
        while positive_instances > 0:
            b1_instance = b1_instances[random.randint(0, len(b1_instances) - 1)]
            b2_instance = b2_instances[random.randint(0, len(b2_instances) - 1)]
            try:
                first_img = os.path.getsize(b1_instance)
                second_img = os.path.getsize(b2_instance)
                if first_img != 0 and second_img != 0:
                    if not ([b1_instance, b2_instance] in false_pairs[word_part]):
                        false_pairs[word_part].append([b1_instance, b2_instance])
                        positive_instances -= 1
            except OSError as e:
                print(e)

    return false_pairs


def create_pairs(root_path, limit, min_instances):
    path = root_path + 'pairs/'
    all_files = [f for f in listdir(path) if isfile(join(path, f))]
    word_part_imgs_b1 = {}
    word_part_imgs_b2 = {}

    for one_file in all_files:
        parts = one_file.split('+')

        first_book = root_path + parts[0].split('-')[0] + '/'
        second_book = root_path + parts[1].split('-')[0] + '/'

        first_dir = parts[0].split(parts[0].split('-')[0] + '-')[1]
        second_dir = parts[1].split(parts[1].split('-')[0] + '-')[1]

        first_dir = first_book + first_dir + '/CCs/'
        second_dir = second_book + second_dir[:-4] + '/CCs/'

        with codecs.open(path + one_file, 'r', "utf-8-sig") as in_fd:
            for line in in_fd:
                line = line.strip().split()
                if line == '':
                    continue
                first_file = first_dir + line[0]
                second_file = second_dir + line[1]
                try:
                    if line[2] in word_part_imgs_b1.keys():
                        word_part_imgs_b1[line[2]].add(first_file)
                    else:
                        word_part_imgs_b1[line[2]] = set()
                        word_part_imgs_b1[line[2]].add(first_file)
                    if line[2] in word_part_imgs_b2.keys():
                        word_part_imgs_b2[line[2]].add(second_file)
                    else:
                        word_part_imgs_b2[line[2]] = set()
                        word_part_imgs_b2[line[2]].add(second_file)
                except OSError as e:
                    print(e)
    print('creating true pairs..')
    true_pairs = create_true_pairs(word_part_imgs_b1, word_part_imgs_b2, min_instances)
    true_pairs = shuffle_and_limit(true_pairs, limit)

    print('forms count =', len(true_pairs))
    print('wordparts =', true_pairs.keys())
    sizes = []
    for key in true_pairs.keys():
        sizes.append(len(true_pairs[key]))
    print('sizes =', sizes)

    print('creating false pairs..')
    false_pairs = create_false_pairs(word_part_imgs_b1, word_part_imgs_b2, true_pairs)
    false_pairs = shuffle_and_limit(false_pairs, limit)

    print('forms count =', len(false_pairs))
    print('wordparts =', false_pairs.keys())
    sizes = []
    for key in true_pairs.keys():
        sizes.append(len(true_pairs[key]))
    print('sizes =', sizes)

    return true_pairs, false_pairs


def prepare_trainset(true_pairs, false_pairs):
    train_pairs = []
    train_y = []

    for key in true_pairs.keys():
        key_true_pairs = true_pairs[key]
        ktp_len = len(key_true_pairs)
        key_false_pairs = false_pairs[key]
        kfp_len = len(key_false_pairs)

        train_pairs.extend(key_true_pairs)
        train_y.extend(itertools.repeat(1, ktp_len))

        train_pairs.extend(key_false_pairs)
        train_y.extend(itertools.repeat(0, kfp_len))

    train = list(zip(train_pairs, train_y))
    np.random.shuffle(train)
    train = np.array(train)

    return train[:, 0], train[:, 1]


def prepare_valid_test(true_pairs, false_pairs):
    test_pairs = []
    test_y = []
    valid_pairs = []
    valid_y = []

    for key in true_pairs.keys():
        key_true_pairs = true_pairs[key]
        ktp_len = len(key_true_pairs)
        key_false_pairs = false_pairs[key]
        kfp_len = len(key_false_pairs)

        test_pairs.extend(key_true_pairs[:int(ktp_len * .5)])
        test_y.extend(itertools.repeat(1, int(ktp_len * .5)))

        test_pairs.extend(key_false_pairs[:int(kfp_len * .5)])
        test_y.extend(itertools.repeat(0, int(kfp_len * .5)))

        valid_pairs.extend(key_true_pairs[int(ktp_len * .5):])
        valid_y.extend(itertools.repeat(1, ktp_len - int(ktp_len * .5)))

        valid_pairs.extend(key_false_pairs[int(kfp_len * .5):])
        valid_y.extend(itertools.repeat(0, kfp_len - int(kfp_len * .5)))

    test = list(zip(test_pairs, test_y))
    np.random.shuffle(test)
    valid = list(zip(valid_pairs, valid_y))
    np.random.shuffle(valid)

    test = np.array(test)
    valid = np.array(valid)

    return test[:, 0], test[:, 1], valid[:, 0], valid[:, 1]


def create_base_network(input_dim):
    seq = Sequential()
    seq.add(Convolution2D(64, (5, 5), padding='same', activation='relu', input_shape=input_dim))
    seq.add(MaxPooling2D(padding='same', pool_size=(2, 2)))
    seq.add(Convolution2D(128, (4, 4), padding='same', activation='relu'))
    seq.add(MaxPooling2D(padding='same', pool_size=(2, 2)))
    seq.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
    seq.add(MaxPooling2D(padding='same', pool_size=(2, 2)))
    seq.add(Convolution2D(512, (2, 2), padding='same', activation='relu'))

    seq.add(Flatten())
    seq.add(Dense(4096, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(4096, activation='relu'))

    return seq


def generate_arrays_from_file(pairs, ys, batch_size=100):
    i = 0
    while True:
        first = []
        second = []
        batch_ys = []
        success = 0
        for pair in pairs[i:]:
            try:
                first_file = misc.imread(pair[0], flatten=True) / 255.
                second_file = misc.imread(pair[1], flatten=True) / 255.

                first.append(np.array(first_file))
                second.append(np.array(second_file))
                batch_ys.append(ys[i])

                success += 1
            except (IOError, OSError, AttributeError) as e:
                print(e)
            finally:
                i += 1
                if success == batch_size:
                    break

        f = np.asarray(first)
        f = f.reshape(f.shape + (1,))
        s = np.asarray(second)
        s = s.reshape(s.shape + (1,))
        if i == (len(pairs) / batch_size) * batch_size:
            i = 0

        yield ([f, s], batch_ys)


def abs_diff_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1


def get_abs_diff(vects):
    x, y = vects
    return K.abs(x - y)


np.random.seed(1337)
g_root_path = '/DATA/majeek/data/0206-0207/combined/'
g_create_train = True
g_create_test_valid = True
g_batch_size = 64
g_limit = 200
g_nb_epoch = 200
g_min_instances = 10

if g_create_train:
    print('preparing training set...')
    g_true_pairs, g_false_pairs = create_pairs(g_root_path + 'train/', g_limit, g_min_instances)
    g_tr_pairs, g_tr_y = prepare_trainset(g_true_pairs, g_false_pairs)
    with open('./trainset.pkl', 'wb') as out_fd:
        pickle.dump([g_tr_pairs, g_tr_y], out_fd)
    print('...done')

if g_create_test_valid:
    print('preparing test/validation sets...')
    g_true_pairs, g_false_pairs = create_pairs(g_root_path + 'test_valid/', g_limit, g_min_instances)
    g_te_pairs, g_te_y, g_va_pairs, g_va_y = prepare_valid_test(g_true_pairs, g_false_pairs)
    with open('./testset.pkl', 'wb') as out_fd:
        pickle.dump([g_te_pairs, g_te_y], out_fd)
    with open('./validset.pkl', 'wb') as out_fd:
        pickle.dump([g_va_pairs, g_va_y], out_fd)
    print('...done')

if not g_create_train:
    with open('./trainset.pkl', 'rb') as in_fd:
        pkl = pickle.load(in_fd)
        g_tr_pairs = pkl[0]
        g_tr_y = pkl[1]

if not g_create_test_valid:
    with open('./testset.pkl', 'rb') as in_fd:
        pkl = pickle.load(in_fd)
        g_te_pairs = pkl[0]
        g_te_y = pkl[1]
    with open('./validset.pkl', 'rb') as in_fd:
        pkl = pickle.load(in_fd)
        g_va_pairs = pkl[0]
        g_va_y = pkl[1]

x = len(g_tr_pairs) % g_batch_size
g_tr_pairs = g_tr_pairs[:len(g_tr_pairs) - x]
g_tr_y = g_tr_y[:len(g_tr_y) - x]

x = len(g_te_pairs) % g_batch_size
g_te_pairs = g_te_pairs[:len(g_te_pairs) - x]
g_te_y = g_te_y[:len(g_te_y) - x]

x = len(g_va_pairs) % g_batch_size
g_va_pairs = g_va_pairs[:len(g_va_pairs) - x]
g_va_y = g_va_y[:len(g_va_y) - x]

print('training size:', len(g_tr_pairs))
print('testing size:', len(g_te_pairs))
print('validation size:', len(g_va_pairs))

input_dim = (83, 69, 1)
g_train = True

base_network = create_base_network(input_dim)
input_a = Input(shape=input_dim)
input_b = Input(shape=input_dim)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

abs_diff = Lambda(get_abs_diff, output_shape=abs_diff_output_shape)([processed_a, processed_b])
flattened_weighted_distance = Dense(1, activation='sigmoid')(abs_diff)

checkpoint = ModelCheckpoint('./bestModel', monitor='val_acc', save_best_only=True, mode='max', verbose=1)

if g_train:
    model = Model(inputs=[input_a, input_b], outputs=flattened_weighted_distance)
    rms = Adam()
    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
    print(model.summary())
    print('training steps=', len(g_tr_pairs) / g_batch_size)
    print('validation steps=', len(g_va_pairs) / g_batch_size)
    model.fit_generator(generate_arrays_from_file(g_tr_pairs, g_tr_y),
                        len(g_tr_pairs) / g_batch_size,
                        epochs=g_nb_epoch,
                        callbacks=[checkpoint],
                        validation_data=generate_arrays_from_file(g_va_pairs, g_va_y),
                        validation_steps=len(g_va_pairs) / g_batch_size)
    model.save(g_root_path + 'model.h5')
else:
    model = load_model(g_root_path + 'bestModel')

# compute final accuracy on training and test sets
tr_score, tr_acc = model.evaluate_generator(generate_arrays_from_file(g_tr_pairs, g_tr_y), len(g_tr_pairs) / g_batch_size)
print('* Accuracy on the training set: {:.2%}'.format(tr_acc))

te_score, te_acc = model.evaluate_generator(generate_arrays_from_file(g_te_pairs, g_te_y), len(g_te_pairs) / g_batch_size)
print('* Accuracy on the test set: {:.2%}'.format(te_acc))

va_score, va_acc = model.evaluate_generator(generate_arrays_from_file(g_va_pairs, g_va_y), len(g_va_pairs) / g_batch_size)
print('* Accuracy on the validation set: {:.2%}'.format(va_acc))