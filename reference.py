class MultivariateGaussian(object):
    def __init__(self, mu, Sigma2, reg_lambda=1e-6):
        assert mu.ndim == 1
        assert Sigma2.ndim == 2
        assert Sigma2.shape[0] == Sigma2.shape[1] == mu.shape[0]
        try:
            self._mu, self._Sigma = mu, la.cholesky(Sigma2 + reg_lambda*np.eye(Sigma2.shape[0]))
        except la.LinAlgError:
            print np.real_if_close(la.eigvals(Sigma2))
            raise
    @property
    def mean(self):
        return self._mu
    @property
    def covariance(self):
        return self._Sigma
    def sample(self):
        xx = np.random.randn(self.mean.shape[0])
        return self.mean + self.covariance.dot(xx)

def se_kernel(a,b,ll,mm):
    return mm*mm*np.exp(-0.5*((a-b)/ll)**2)

def rbf_covariance(x1,x2,ll=1.,mm=1.):
    return covariance_matrix(x1,x2,se_kernel, (ll,mm))
    
def covariance_matrix(x1,x2, kernel, hyperparams):
    """x1 and x2 are both vectors; we return the covariance matrix 
    of the two, under the Gaussian covariance kernel. I call it rbf here to
    avoid confusion with the Gaussian in 'Gaussian Processes'
    """
    answer = np.zeros((len(x1), len(x2)))
    for ii in xrange(len(x1)):
        for jj in xrange(len(x2)):
            answer[ii,jj] = kernel(x1[ii], x2[jj], *hyperparams)
    return answer

def symmetric(xx):
    return np.allclose(xx,xx.T,rtol=1e-4,atol=1e-5)

class GPRegression(object):
    def __init__(self, kernel=se_kernel):
        self.kernel = kernel
        
    def fit(self, training_X, training_y, hyperparams, noise=1e-4):
        self.ntraining = training_y.shape[0]
        self.training_X = training_X
        self.training_y = training_y
        return self.refit(hyperparams,noise)
    
    def refit(self, hyperparams, noise):
        self.hyperparams = hyperparams
        noise_vec = noise*np.ones(self.ntraining)
        self.K = self.covariance_matrix(self.training_X) + np.diag(noise_vec)
        self.K_inv_y = la.solve(self.K, self.training_y)
        assert symmetric(self.K)
        self.K_inv = None
        
    def covariance_matrix(self, x1,x2=None):
        if x2 is None: x2=x1
        return covariance_matrix(x1,x2,self.kernel,self.hyperparams)
    
    def prior_distribution(self, test_X, hyperparams):
        self.hyperparams = hyperparams
        K_ss = self.covariance_matrix(test_X)
        assert symmetric(K_ss)
        mean = np.zeros(test_X.shape[0])
        return MultivariateGaussian(mean, K_ss)
    
    def posterior_distribution(self, test_X):
        if self.K_inv is None:
            self.K_inv = la.inv(self.K)
            assert symmetric(self.K_inv)
        K_ss = self.covariance_matrix(test_X)
        K_s = self.covariance_matrix(test_X, self.training_X)
        mean = K_s.dot(self.K_inv_y)
        covar = K_ss - K_s.dot(self.K_inv).dot(K_s.T)
        assert symmetric(K_ss) and symmetric(covar)
        return MultivariateGaussian(mean, covar)
    
    def posterior_mean(self, test_X):
        K_s = self.covariance_matrix(test_X, self.training_X)
        return K_s.dot(self.K_inv_y)
    
    def residual(self, test_X, test_y):
        post_mean = self.posterior_mean(test_X)
        residual = post_mean - test_y
        return residual

def linear_kernel(x,y,c,m):
    return m*m*(x-c)*(y-c)

def quadratic_kernel(x,y,c1,c2,m):
    return m*m*(x-c1)*(y-c1)*(x-c2)*(y-c2)

def inverse_r_kernel(x,y,c,m):
    return m*m*(1./x-c)*(1./y-c)

def inverse_r_times_se_kernel(x,y,c,l, m):
    return m*m*((1./x-c)*(1./y-c) * np.exp(-(x-y)**2/l**2))
