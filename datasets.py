import numpy as np
import cPickle as pickle
import scipy.io


def load_mnist(data_file="/u/wgrathwohl/relaxed-rebar/data/mnist_salakhutdinov_07-19-2017.pkl"):
    with open(data_file, 'r') as f:
         (tr, _), (va, _), (te, _) = pickle.load(f)
         return tr, va, te


def load_omniglot(data_file='/u/wgrathwohl/relaxed-rebar/data/omniglot_07-19-2017.mat'):
  """Reads in Omniglot images.

  Args:
    binarize: whether to use the fixed binarization

  Returns:
    x_train: training images
    x_valid: validation images
    x_test: test images

  """
  n_validation=1345

  def reshape_data(data):
    return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')

  omni_raw = scipy.io.loadmat(data_file)

  train_data = reshape_data(omni_raw['data'].T.astype('float32'))
  test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))

  # Binarize the data with a fixed seed
  np.random.seed(5)
  train_data = (np.random.rand(*train_data.shape) < train_data).astype(float)
  test_data = (np.random.rand(*test_data.shape) < test_data).astype(float)

  shuffle_seed = 123
  permutation = np.random.RandomState(seed=shuffle_seed).permutation(train_data.shape[0])
  train_data = train_data[permutation]

  x_train = train_data[:-n_validation]
  x_valid = train_data[-n_validation:]
  x_test = test_data

  return x_train, x_valid, x_test