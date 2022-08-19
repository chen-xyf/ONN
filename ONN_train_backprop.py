import time
import numpy as np
from termcolor import colored
from ONN_class import MyONN
from sklearn.datasets import make_classification
from datetime import datetime
import os

n, m, k = 3, 10, 3

batch_size = 10
num_batches = 20
num_epochs = 20
lr = 0.01

slm_arr = np.random.normal(0, 0.5, (n, m))
slm_arr = np.clip(slm_arr, -1, 1)
slm_arr = (slm_arr*64).astype(np.int)/64

slm2_arr = np.random.normal(0, 0.5, (k, m))
slm2_arr = np.clip(slm2_arr, -1, 1)
slm2_arr = (slm2_arr*64).astype(np.int)/64

x, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0,
                           scale=1.0, shuffle=True, random_state=6)

y_onehot = np.zeros((500, 3))

for i in range(500):
    y_onehot[i, int(y[i])] = 1

trainy = y_onehot[:400, :].copy()
testy = y_onehot[400:, :].copy()

xnorm = x.copy()
xnorm[:, 0] -= xnorm[:, 0].min()
xnorm[:, 0] *= 1./xnorm[:, 0].max()
xnorm[:, 1] -= xnorm[:, 1].min()
xnorm[:, 1] *= 1./xnorm[:, 1].max()

trainx = xnorm[:400, :].copy()
testx = xnorm[400:, :].copy()

today_date = datetime.today().strftime('%Y-%m-%d')
save_folder = 'D:/ONN_backprop/'+today_date+'/run_3/'

if os.path.exists(save_folder+'NOTES.txt'):
    print('Folder already exists!')
    raise RuntimeError
else:
    with open(save_folder+'NOTES.txt', mode="w") as f:
        f.write("OPTICAL TRAINING")

np.save(save_folder + 'trainx.npy', trainx)
np.save(save_folder + 'trainy.npy', trainy)
np.save(save_folder + 'testx.npy', testx)
np.save(save_folder + 'testy.npy', testy)

all_params = (n, m, k, batch_size, num_batches, num_epochs, lr)
np.save(save_folder + 'all_params.npy', all_params)

onn = MyONN(num_batches=num_batches, num_epochs=num_epochs,
            w1_0=slm_arr[1:, :], w2_0=slm2_arr, b1_0=slm_arr[0, :],
            lr=lr, ctrl_vars=(n, m, k, batch_size),
            save_folder=save_folder,
            trainx=trainx, testx=testx, trainy=trainy, testy=testy)

for epoch in range(num_epochs):

    t0 = time.perf_counter()

    if epoch == 0:
        onn.run_calibration(initial=True)
    # else:
    #     onn.run_calibration(initial=True)

    rng = np.random.default_rng(epoch)
    epoch_rand_indxs = np.arange(onn.trainx.shape[0])
    rng.shuffle(epoch_rand_indxs)
    onn.batch_indxs_list = [epoch_rand_indxs[i * onn.batch_size: (i + 1) * onn.batch_size]
                            for i in range(onn.num_batches)]

    for batch in range(num_batches):
        success = onn.run_batch(batch)
        if success:
            onn.save_batch(epoch, batch)
            onn.graph_batch(epoch, batch)
            dt = onn.loop_clock.tick()
            print(colored(f'batch time: {dt:.3f}', 'yellow'))
            print()

    onn.run_validation(epoch)

    t1 = time.perf_counter()
    epoch_time = t1 - t0

    print('\n######################################################################')
    print(colored('epoch {}, time : {}, accuracy : {:.2f}, final loss : {:.2f}'
                  .format(epoch, epoch_time, onn.accs[-1], onn.loss[-1]), 'green'))
    print('######################################################################\n')

print('done')
