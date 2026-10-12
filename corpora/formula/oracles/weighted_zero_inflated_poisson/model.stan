// generated with brms 2.23.2
functions {
  /* zero-inflated poisson log-PDF of a single response
   * Args:
   *   y: the response value
   *   lambda: mean parameter of the poisson distribution
   *   zi: zero-inflation probability
   * Returns:
   *   a scalar to be added to the log posterior
   */
  real zero_inflated_poisson_lpmf(int y, real lambda, real zi) {
    if (y == 0) {
      return log_sum_exp(bernoulli_lpmf(1 | zi),
                         bernoulli_lpmf(0 | zi) +
                         poisson_lpmf(0 | lambda));
    } else {
      return bernoulli_lpmf(0 | zi) +
             poisson_lpmf(y | lambda);
    }
  }
  /* zero-inflated poisson log-PDF of a single response
   * logit parameterization of the zero-inflation part
   * Args:
   *   y: the response value
   *   lambda: mean parameter of the poisson distribution
   *   zi: linear predictor for zero-inflation part
   * Returns:
   *   a scalar to be added to the log posterior
   */
  real zero_inflated_poisson_logit_lpmf(int y, real lambda, real zi) {
    if (y == 0) {
      return log_sum_exp(bernoulli_logit_lpmf(1 | zi),
                         bernoulli_logit_lpmf(0 | zi) +
                         poisson_lpmf(0 | lambda));
    } else {
      return bernoulli_logit_lpmf(0 | zi) +
             poisson_lpmf(y | lambda);
    }
  }
  /* zero-inflated poisson log-PDF of a single response
   * log parameterization for the poisson part
   * Args:
   *   y: the response value
   *   eta: linear predictor for poisson distribution
   *   zi: zero-inflation probability
   * Returns:
   *   a scalar to be added to the log posterior
   */
  real zero_inflated_poisson_log_lpmf(int y, real eta, real zi) {
    if (y == 0) {
      return log_sum_exp(bernoulli_lpmf(1 | zi),
                         bernoulli_lpmf(0 | zi) +
                         poisson_log_lpmf(0 | eta));
    } else {
      return bernoulli_lpmf(0 | zi) +
             poisson_log_lpmf(y | eta);
    }
  }
  /* zero-inflated poisson log-PDF of a single response
   * log parameterization for the poisson part
   * logit parameterization of the zero-inflation part
   * Args:
   *   y: the response value
   *   eta: linear predictor for poisson distribution
   *   zi: linear predictor for zero-inflation part
   * Returns:
   *   a scalar to be added to the log posterior
   */
  real zero_inflated_poisson_log_logit_lpmf(int y, real eta, real zi) {
    if (y == 0) {
      return log_sum_exp(bernoulli_logit_lpmf(1 | zi),
                         bernoulli_logit_lpmf(0 | zi) +
                         poisson_log_lpmf(0 | eta));
    } else {
      return bernoulli_logit_lpmf(0 | zi) +
             poisson_log_lpmf(y | eta);
    }
  }
  // zero-inflated poisson log-CCDF and log-CDF functions
  real zero_inflated_poisson_lccdf(int y, real lambda, real zi) {
    return bernoulli_lpmf(0 | zi) + poisson_lccdf(y | lambda);
  }
  real zero_inflated_poisson_lcdf(int y, real lambda, real zi) {
    return log1m_exp(zero_inflated_poisson_lccdf(y | lambda, zi));
  }
}
data {
  int<lower=1> N;  // total number of observations
  array[N] int Y;  // response variable
  vector<lower=0>[N] weights;  // model weights
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
  real<lower=0,upper=1> zi;  // zero-inflation probability
}
transformed parameters {
  // prior contributions to the log posterior
  real lprior = 0;
  lprior += student_t_lpdf(Intercept | 3, -0.8, 2.5);
  lprior += beta_lpdf(zi | 1, 1);
}
model {
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N] mu = rep_vector(0.0, N);
    mu += Intercept + Xc * b;
    for (n in 1:N) {
      target += weights[n] * (zero_inflated_poisson_log_lpmf(Y[n] | mu[n], zi));
    }
  }
  // priors including constants
  target += lprior;
}
generated quantities {
  // actual population-level intercept
  real b_Intercept = Intercept - dot_product(means_X, b);
}

