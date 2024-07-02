import numpy as np
######### tfp
import tensorflow as tf
# do not use gpu, because it is even slower
tf.config.set_visible_devices([], 'GPU')
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from scipy.special import softmax

class Glass:
    def __init__(self, shrinkage_factor=0, dtype = tf.float32) -> None:
        self.shrinkage_factor, self.dtype = shrinkage_factor, dtype
    
    def process_data(self, X_train: np.ndarray, y_train: np.ndarray):
        self.nchannel, self.nT = X_train.shape[2:]
        self.X_train = tf.constant(X_train, dtype = self.dtype)
        self.y_train = tf.constant(y_train, dtype = self.dtype)
        def jointmodel():
            sigma = yield tfd.Sample(tfd.HalfCauchy(0.0, tf.cast(1, self.dtype)), 2)
            # weights
            beta = yield tfb.Cumsum()(tfd.Sample(tfd.Normal(0.0, 1.0), (2, self.nT)))
            beta = sigma[:, None] * beta
            beta = tfp.math.soft_threshold(beta, self.shrinkage_factor)
            probs = yield tfd.RelaxedOneHotCategorical(1, logits=tf.zeros([self.nchannel, 3]))
            weights = yield tfd.Sample(tfd.Normal(0.0, 1.0), [self.nchannel, 2])
            weights = tf.linalg.normalize(weights, axis=0)[0]
            beta = tf.linalg.matmul(probs[:,1:]*weights, beta)
            logits = tf.linalg.tensordot(self.X_train, beta, axes = [[2, 3], [0, 1]])
            y = yield tfd.Multinomial(logits = logits, total_count = 1)
        self.joint = tfd.JointDistributionCoroutineAutoBatched(jointmodel)

    def mfvb(self, num_steps=2000, sample_size=10, importance_sample_size=10, learning_rate=0.05, seed=1):
        self.posterior = tfd.JointDistributionSequentialAutoBatched([
            tfd.LogNormal(
                tf.Variable(tf.zeros(2, dtype = self.dtype) - 3),
                tfp.util.TransformedVariable(0.1 * tf.ones(2, dtype = self.dtype), bijector = tfb.Softplus())),
            tfd.Normal(
                tf.Variable(tf.random.normal((2, self.nT), stddev=3*self.shrinkage_factor, dtype = self.dtype), 
                            dtype = self.dtype),
                tfp.util.TransformedVariable(0.01 * tf.ones((2, self.nT), dtype = self.dtype), 
                                             bijector = tfb.Softplus())),
            tfd.RelaxedOneHotCategorical(0.5, logits=tf.Variable(tf.zeros((self.nchannel, 3)))),
            tfd.Normal(tf.Variable(tf.zeros([self.nchannel, 2], dtype = self.dtype), dtype = self.dtype),
                       tfp.util.TransformedVariable(tf.ones([self.nchannel, 2], dtype = self.dtype), 
                                                    dtype = self.dtype,
                                                    bijector = tfb.Softplus()))
        ])

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        tf.random.set_seed(seed)
        self.losses = list(tfp.vi.fit_surrogate_posterior(
            self.loglik, 
            self.posterior,
            optimizer = optimizer,
            num_steps = num_steps, 
            sample_size = sample_size,
            importance_sample_size=importance_sample_size))
        self.losses = [float(x) for x in self.losses]
        
        self.samples = self.posterior.sample(5000)
        self.process_samples()

    @property
    def median_sample(self):
        return [np.median(x, axis = 0) for x in self.samples]
    
    def loglik(self, *args):
        return self.joint.log_prob(*args, self.y_train)
        
    def process_samples(self):
        sigmas, self.betagMats, self.probs, self.weights = [np.array(x) for x in self.samples]
        l2 = np.sqrt(np.sum(np.square(self.weights), 1))
        self.weights = self.weights/l2[:, None, :]
        self.effective_weights = self.probs[:,:,1:]*self.weights
        self.weight = self.weights.mean(axis = 0)
        self.effective_weight = self.effective_weights.mean(axis = 0)
        self.betagMats = sigmas[:, :, None] * self.betagMats
        self.betagMats = np.array(tfp.math.soft_threshold(self.betagMats, self.shrinkage_factor))
        self.betagMat = np.median(self.betagMats, axis=0)
        self.betaMats = np.matmul(self.effective_weights, self.betagMats)
        self.betaMat = np.median(self.betaMats, axis=0)
    
    def predict_prob(self, newX: np.ndarray, method = 'median'):
        if method == 'median':
            logodds = np.tensordot(newX, self.betaMat, axes=[[2, 3], [0, 1]])
            probs = softmax(logodds, axis=1)
        elif method == 'vote':
            logodds = np.tensordot(newX, self.betaMats, axes=[[2, 3], [1, 2]])
            probs = softmax(logodds, axis=1)
            probs = probs.mean(axis=2)
        return probs
