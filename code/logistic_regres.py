
import os
import numpy as np
import sklearn.metrics as sm
from scipy.special import expit
from matplotlib import pyplot as plt


def safe_log(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))

def loss_fn(X, w, y):
    z = np.matmul(X, w)
    sigm_z = expit(z)
    retu = y * safe_log(sigm_z) + (1 - y) * safe_log(1 - sigm_z)
    retu = -1*retu.sum()
    return retu

def gradient(X, w, y):
    z = np.matmul(X, w)
    sigm = expit(z)
    grad = (y - sigm)*X
    summat = np.sum(grad, axis=0)
    retu = np.reshape(summat, w.shape)
    return retu

def weights_save(w, l, e):
    dir = "./output/logistic_basic/"
    np.save(dir + "weights_" + str(e) + ".npy", w)
    np.save(dir + "loss_" + str(e) + ".npy", l)

def perf_save(l, acc, f1):
    dir = "./output/logistic_basic/"
    np.save(dir + "val_loss" + ".npy", l)
    np.save(dir + "val_acc" + ".npy", acc)
    np.save(dir + "val_f1" + ".npy", f1)

def main():
    image_dir = "./data/training/images/"
    mask_dir = "./data/training/masks/"
    val_image_dir = "./data/validation/images/"
    val_mask_dir = "./data/validation/masks/"

    w = np.random.randn(4, 1)
    a = 0.01
    ep = 45
    sum_l = 0
    avg_l = 0
    step = 0

    for e in range(ep):
        print("In epoch:", e + 1)
        for f_name in os.listdir(image_dir):
            X = np.load(os.path.join(image_dir, f_name))
            X = np.reshape(X, (X.shape[0] * X.shape[1], 3))
            one = np.ones((X.shape[0], 1))
            X = np.concatenate((one, X), 1)
            y = np.load(os.path.join(mask_dir, f_name))
            y = np.reshape(y, (y.shape[0] * y.shape[1], 1))
            l = loss_fn(X, w, y)
            sum_l += l
            step += 1
            avg_l = sum_l / step

            w = w + a * gradient(X, w, y)

        print("Loss after Epoch: {} is {}".format(e + 1, avg_l))
        weights_save(w, avg_l, e + 1)

    avg_val_l = 0
    avg_val_acc = 0
    avg_f1 = 0
    step = 0
    sum_l = 0
    sum_acc = 0
    sum_f1 = 0

    for f_name in os.listdir(val_image_dir):
        X_val = np.load(os.path.join(val_image_dir, f_name))
        X_val = np.reshape(X_val, (X_val.shape[0] * X_val.shape[1], 3))
        one = np.ones((X_val.shape[0], 1))
        X_val = np.concatenate((one, X_val), 1)
        y_val = np.load(os.path.join(val_mask_dir, f_name))
        y_val = np.reshape(y_val, (y_val.shape[0] * y_val.shape[1], 1))

        y_pd = np.matmul(X_val, w) >= 0
        y_pd = y_pd.astype(np.uint8)

        # Metrics calculated here
        l = loss(X_val, y_val, w)
        step += 1
        sum_l += l
        avg_val_l = sum_l / step
        acc = sm.accuracy_score(y_val, y_pd)
        sum_acc += acc
        avg_val_acc = sum_acc / step
        f1 = sm.f1_score(y_val, y_pd)
        sum_f1 += f1
        avg_f1 = sum_f1 / step

    perf_save(avg_val_l, avg_val_acc, avg_f1)

if __name__ == "__main__":
    main()
