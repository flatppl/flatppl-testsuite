// generated with brms 2.23.2
functions {
  /* zero-inflated beta log-PDF of a single response
   * Args:
   *   y: the response value
   *   mu: mean parameter of the beta distribution
   *   phi: precision parameter of the beta distribution
   *   zi: zero-inflation probability
   * Returns:
   *   a scalar to be added to the log posterior
   */
   real zero_inflated_beta_lpdf(real y, real mu, real phi, real zi) {
     row_vector[2] shape = [mu * phi, (1 - mu) * phi];
     if (y == 0) {
       return bernoulli_lpmf(1 | zi);
     } else {
       return bernoulli_lpmf(0 | zi) +
              beta_lpdf(y | shape[1], shape[2]);
     }
   }
  /* zero-inflated beta log-PDF of a single response
   * logit parameterization of the zero-inflation part
   * Args:
   *   y: the response value
   *   mu: mean parameter of the beta distribution
   *   phi: precision parameter of the beta distribution
   *   zi: linear predictor for zero-inflation part
   * Returns:
   *   a scalar to be added to the log posterior
   */
   real zero_inflated_beta_logit_lpdf(real y, real mu, real phi, real zi) {
     row_vector[2] shape = [mu * phi, (1 - mu) * phi];
     if (y == 0) {
       return bernoulli_logit_lpmf(1 | zi);
     } else {
       return bernoulli_logit_lpmf(0 | zi) +
              beta_lpdf(y | shape[1], shape[2]);
     }
   }
  // zero-inflated beta log-CCDF and log-CDF functions
  real zero_inflated_beta_lccdf(real y, real mu, real phi, real zi) {
    row_vector[2] shape = [mu * phi, (1 - mu) * phi];
    return bernoulli_lpmf(0 | zi) + beta_lccdf(y | shape[1], shape[2]);
  }
  real zero_inflated_beta_lcdf(real y, real mu, real phi, real zi) {
    return log1m_exp(zero_inflated_beta_lccdf(y | mu, phi, zi));
  }
}
data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
  int<lower=1> Kc;  // number of population-level effects after centering
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
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
  real<lower=0> phi;  // precision parameter
  real<lower=0,upper=1> zi;  // zero-inflation probability
}
transformed parameters {
  // prior contributions to the log posterior
  real lprior = 0;
  lprior += student_t_lpdf(Intercept | 3, 0, 2.5);
  lprior += gamma_lpdf(phi | 0.01, 0.01);
  lprior += beta_lpdf(zi | 1, 1);
}
model {
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N] mu = rep_vector(0.0, N);
    mu += Intercept + Xc * b;
    mu = inv_logit(mu);
    for (n in 1:N) {
      target += zero_inflated_beta_lpdf(Y[n] | mu[n], phi, zi);
    }
  }
  // priors including constants
  target += lprior;
}
generated quantities {
  // actual population-level intercept
  real b_Intercept = Intercept - dot_product(means_X, b);
}

