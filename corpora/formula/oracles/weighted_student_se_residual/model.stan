// generated with brms 2.23.2
functions {
  /* compute the logm1 link
   * Args:
   *   p: a positive scalar
   * Returns:
   *   a scalar in (-Inf, Inf)
   */
   real logm1(real y) {
     return log(y - 1.0);
   }
  /* compute the logm1 link (vectorized)
   * Args:
   *   p: a positive vector
   * Returns:
   *   a vector in (-Inf, Inf)
   */
   vector logm1(vector y) {
     return log(y - 1.0);
   }
  /* compute the inverse of the logm1 link
   * Args:
   *   y: a scalar in (-Inf, Inf)
   * Returns:
   *   a positive scalar
   */
   real expp1(real y) {
     return exp(y) + 1.0;
   }
  /* compute the inverse of the logm1 link (vectorized)
   * Args:
   *   y: a vector in (-Inf, Inf)
   * Returns:
   *   a positive vector
   */
   vector expp1(vector y) {
     return exp(y) + 1.0;
   }
}
data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  vector<lower=0>[N] weights;  // model weights
  vector<lower=0>[N] se;  // known sampling error
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
  int<lower=1> Kc;  // number of population-level effects after centering
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
  vector<lower=0>[N] se2 = square(se);
  matrix[N, Kc] Xc;  // centered version of X without an intercept
  vector[Kc] means_X;  // column means of X before centering
  for (i in 2:K) {
    means_X[i - 1] = mean(X[, i]);
    Xc[, i - 1] = X[, i] - means_X[i - 1];
  }
}
parameters {
  vector[Kc] b;  // regression coefficients
  real Intercept;  // temporary intercept for centered predictors
  real<lower=0> sigma;  // dispersion parameter
}
transformed parameters {
  real<lower=1> nu;  // degrees of freedom or shape
  // prior contributions to the log posterior
  real lprior = 0;
  nu = 5;
  lprior += normal_lpdf(b | 0, 1);
  lprior += normal_lpdf(Intercept | 0, 2);
  lprior += normal_lpdf(sigma | 0, 1)
    - 1 * normal_lccdf(0 | 0, 1);
}
model {
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N] mu = rep_vector(0.0, N);
    mu += Intercept + Xc * b;
    for (n in 1:N) {
      target += weights[n] * (student_t_lpdf(Y[n] | nu, mu[n], sqrt(square(sigma) + se2[n])));
    }
  }
  // priors including constants
  target += lprior;
}
generated quantities {
  // actual population-level intercept
  real b_Intercept = Intercept - dot_product(means_X, b);
}

