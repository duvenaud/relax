import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle

fn = '100_samples.pkl'
with open(fn, 'r') as f:
    results = pickle.load(f)
    
reinf_fs, reinf_cs, rep_cs, zlaxs = results
    
min_val = -3#np.min(np.concatenate([reinf_fs, reinf_cs, rep_cs, zlaxs]))
max_val = 3#np.max(np.concatenate([reinf_fs, reinf_cs, rep_cs, zlaxs]))
print("++++++++++++++++++++++++++")
print(np.min(reinf_fs), np.max(reinf_fs))
print(np.min(reinf_cs), np.max(reinf_cs))
print(np.min(reinf_fs-reinf_cs), np.max(reinf_fs-reinf_cs))
print(np.min(rep_cs), np.max(rep_cs))
print(np.min(zlaxs), np.max(zlaxs))
print("reinforce_f variance = {}".format(np.log(reinf_fs.var())))
print("reinforce_c variance = {}".format(np.log(reinf_cs.var())))
print("reparam_c  variance  = {}".format(np.log(rep_cs.var())))
print("zlaxs variance       = {}".format(np.log(zlaxs.var())))
print("++++++++++++++++++++++++++")

matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

plt.figure(1, figsize=(20,20))

plt1 = plt.subplot(4, 4, 2)
plt.hist(reinf_fs, 50, range=(min_val, max_val), normed=1, facecolor='g', alpha=0.75)
plt.xlim(min_val, max_val)
plt.title(r'$\hat g_{\text{REINFORCE}}[f]$', y=1.025)
plt.xlabel('unbiased \nhigh variance')
plt.ylabel('=', rotation=0, size=40)
plt1.yaxis.set_label_coords(-0.13,0.4)
plt1.axes.yaxis.set_ticks([])

plt2 = plt.subplot(4, 4, 3)
plt.hist(reinf_cs, 50, range=(min_val, max_val), normed=1, facecolor='g', alpha=0.75)
plt.xlim(min_val, max_val)
plt.title(r'$\hat g_{\text{REINFORCE}}[c_\phi]$', y=1.025)
plt.xlabel('biased \nhigh variance')
plt.ylabel(r'-', rotation=0, size=60)
plt2.yaxis.set_label_coords(-0.13,0.35)
plt2.axes.yaxis.set_ticks([])

plt3 = plt.subplot(4, 4, 4)
plt.hist(rep_cs, 50, range=(min_val, max_val), normed=1, facecolor='g', alpha=0.75)
plt.xlim(min_val, max_val)
plt.title(r'$\hat g_{\text{reparam}}[c_\phi]$', y=1.025)
plt3.yaxis.set_label_coords(-0.125,0.4)
plt.xlabel('biased \nlow variance')
plt.ylabel('+', rotation=0, size=40)

plt3.axes.yaxis.set_ticks([])

plt4 = plt.subplot(4, 4, 1)
plt.hist(zlaxs, 50, range=(min_val, max_val), normed=1, facecolor='g', alpha=0.75)
plt.xlim(min_val, max_val)
plt.title(r'$\hat g_{\text{LAX}}$', y=1.025)
plt.xlabel('unbiased \nlow variance')
plt4.axes.yaxis.set_ticks([])


plt.tight_layout()
plt.savefig('./10k_mnist_vae_grad_hist.pdf', format="pdf", bbox_inches='tight')
plt.show()


