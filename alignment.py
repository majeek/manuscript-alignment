import os
import sys
import numpy as np
from scipy import misc
from munkres import Munkres
from keras.models import load_model


os.environ["CUDA_VISIBLE_DEVICES"] = ""


def get_annotation(img_idx, manuscript_path):
    with open(manuscript_path, 'r') as in_f:
        for i in range(0, img_idx):
            value = in_f.readline().decode('utf-8')
        k, v = value.split('\t')
    return v.rstrip()


def find_best_fit(predictions, b1, root1, page1, b2, root2, page2):
    m = Munkres()
    idx = m.compute(1 - predictions)
    total = 0
    for r, c in idx:
        v = predictions[r][c]
        total += v
        annotation1 = get_annotation(b1 + r, root1 + page1 + page1 + '.txt')
        annotation2 = get_annotation(b2 + c, root2 + page2 + page2 + '.txt')
        print('(R %s [%d], %s [%d]) -> %f' % (annotation1, b1 + r - 1, annotation2, b2 + c - 1, v))

    print('total cost: %f' % total)
    return idx


def alignment_iteration(img_idx_manuscript1, img_idx_manuscript2, window_size, model, path1, path2):

    all_paths = []
    for idx1 in range(img_idx_manuscript1, img_idx_manuscript1 + window_size):
        for idx2 in range(img_idx_manuscript2, img_idx_manuscript2 + window_size):
            all_paths.append([path1 + str(idx1) + '.png', path2 + str(idx2) + '.png'])
    print('loading image pairs... for indexes %d, %d' % (img_idx_manuscript1, img_idx_manuscript2))
    first = []
    second = []
    for pair in all_paths:
        first_file = misc.imread(pair[0], flatten=True) / 255.
        second_file = misc.imread(pair[1], flatten=True) / 255.
        first.append(np.array(first_file))
        second.append(np.array(second_file))

    f = np.asarray(first)
    f = f.reshape(f.shape + (1,))
    s = np.asarray(second)
    s = s.reshape(s.shape + (1,))

    sys.stdout.write('predicting... ')
    predicts = model.predict([f, s])
    predicts = np.array(predicts)

    print('done')

    for a, pred in zip(all_paths, predicts):
        a1 = a[0].split('/').pop()
        a2 = a[1].split('/').pop()
        print(a1, a2, pred[0])

    predicts = np.reshape(predicts, (window_size, window_size))
    return predicts


def align_manuscripts_page(root1, path1, page1, begin1, root2, path2, page2, begin2, last_index, window_size, model_path):

    print('loading model...')
    model = load_model(model_path)

    results = {}
    curr1 = begin1
    curr2 = begin2
    while curr1 < last_index:
        p = alignment_iteration(curr1, curr2, window_size, model, path1, path2)
        curr1 += 1
        curr2 += 1
        indexes = find_best_fit(p, curr1, root1, page1, curr2, root2, page2)

        for row, column in indexes:
            value = p[row][column]
            key1 = curr1 + row
            key2 = curr2 + column
            if (key1, key2) in results:
                results[(key1, key2)] += value
            else:
                results[(key1, key2)] = value

    final_results = []
    picks = []
    print('total picks =', len(picks))
    for i in range(begin1 + 1, begin1 + window_size + last_index - 1):
        picks.append([(key, results[key]) for key in results if key[0] == i])

    print(picks)
    for pick in picks:
        print(pick)
        rez = max(pick, key=lambda x: x[1])
        final_results.append(rez)

    success = 0.
    total = 0
    for r in final_results:
        val1 = get_annotation(r[0][0], root1 + page1 + page1 + '.txt')
        val2 = get_annotation(r[0][1], root2 + page2 + page2 + '.txt')
        confidence = r[1]
        if confidence < 1.0:
            print('LOW CONFIDENCE, %d, %d, %s, %s, %s' % (r[0][0], r[0][1], r[1], val1, val2))
            continue
        elif val1 == val2:
            print('success, %d, %d, %s, %s, %s' % (r[0][0], r[0][1], r[1], val1, val2))
            success += 1
            total += 1
        else:
            print('ERROR, %d, %d, %s, %s, %s' % (r[0][0], r[0][1], r[1], val1, val2))
            total += 1

    accuracy = success/total*100

    return accuracy


g_window_size = 7
g_model_path = 'best_model'
g_subword_count = 140

g_begin1 = 1
g_root1 = '/DATA/majeek/data/multiWriter/0206'
g_page1 = '/004-2'
g_path1 = g_root1 + g_page1 + '/CCs/0206-004-2-'

g_begin2 = 144
g_root2 = '/DATA/majeek/data/multiWriter/0207'
g_page2 = '/003-1'
g_path2 = g_root2 + g_page2 + '/CCs/0207-003-1-'


g_accuracy = align_manuscripts_page(g_root1, g_path1, g_page1, g_begin1,
                                    g_root2, g_path2, g_page2, g_begin2,
                                    g_subword_count, g_window_size, g_model_path)
print('success rate = %2f' % g_accuracy)
