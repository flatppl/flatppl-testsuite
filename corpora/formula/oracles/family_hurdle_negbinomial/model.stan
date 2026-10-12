// generated with brms 2.23.2
functions {
  /* hurdle negative binomial log-PDF of a single response
   * Args:
   *   y: the response value
   *   mu: mean parameter of negative binomial distribution
   *   phi: shape parameter of negative binomial distribution
   *   hu: hurdle probability
   * Returns:
   *   a scalar to be added to the log posterior
   */
  real hurdle_neg_binomial_lpmf(int y, real mu, real phi, real hu) {
    if (y == 0) {
      return bernoulli_lpmf(1 | hu);
    } else {
      return bernoulli_lpmf(0 | hu) +
             neg_binomial_2_lpmf(y | mu, phi) -
             log1m((phi / (mu + phi))^phi);
    }
  }
  /* hurdle negative binomial log-PDF of a single response
   * logit parameterization for the hurdle part
   * Args:
   *   y: the response value
   *   mu: mean parameter of negative binomial distribution
   *   phi: phi parameter of negative binomial distribution
   *   hu: linear predictor of hurdle part
   * Returns:
   *   a scalar to be added to the log posterior
   */
  real hurdle_neg_binomial_logit_lpmf(int y, real mu, real phi, real hu) {
   if (y == 0) {
     return bernoulli_logit_lpmf(1 | hu);
   } else {
     return bernoulli_logit_lpmf(0 | hu) +
            neg_binomial_2_lpmf(y | mu, phi) -
            log1m((phi / (mu + phi))^phi);
   }
  }
  /* hurdle negative binomial log-PDF of a single response
   * log parameterization for the negative binomial part
   * Args:
   *   y: the response value
   *   eta: linear predictor for negative binomial distribution
   *   phi phi parameter of negative binomial distribution
   *   hu: hurdle probability
   * Returns:
   *   a scalar to be added to the log posterior
   */
  real hurdle_neg_binomial_log_lpmf(int y, real eta, real phi, real hu) {
    if (y == 0) {
      return bernoulli_lpmf(1 | hu);
    } else {
      return bernoulli_lpmf(0 | hu) +
             neg_binomial_2_log_lpmf(y | eta, phi) -
             log1m((phi / (exp(eta) + phi))^phi);
    }
  }
  /* hurdle negative binomial log-PDF of a single response
   * log parameterization for the negative binomial part
   * logit parameterization for the hurdle part
   * Args:
   *   y: the response value
   *   eta: linear predictor for negative binomial distribution
   *   phi: phi parameter of negative binomial distribution
   *   hu: linear predictor of hurdle part
   * Returns:
   *   a scalar to be added to the log posterior
   */
  real hurdle_neg_binomial_log_logit_lpmf(int y, real eta, real phi, real hu) {
    if (y == 0) {
      return bernoulli_logit_lpmf(1 | hu);
   } else {
      return bernoulli_logit_lpmf(0 | hu) +
             neg_binomial_2_log_lpmf(y | eta, phi) -
             log1m((phi / (exp(eta) + phi))^phi);
    }
  }
  // hurdle negative binomial log-CCDF and log-CDF functions
  real hurdle_neg_binomial_lccdf(int y, real mu, real phi, real hu) {
    return bernoulli_lpmf(0 | hu) + neg_binomial_2_lccdf(y | mu, phi) -
           log1m((phi / (mu + phi))^phi);
  }
  real hurdle_neg_binomial_lcdf(int y, real mu, real phi, real hu) {
    return log1m_exp(hurdle_neg_binomial_lccdf(y | mu, phi, hu));
  }
}
data {
  int<lower=1> N;  // total number of observations
  array[N] int Y;  // response variable
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
  real<lower=0> shape;  // shape parameter
  real<lower=0,upper=1> hu;  // hurdle probability
}
transformed parameters {
  // prior contributions to the log posterior
  real lprior = 0;
  lprior += student_t_lpdf(Intercept | 3, -0.8, 2.5);
  lprior += inv_gamma_lpdf(shape | 0.4, 0.3);
  lprior += beta_lpdf(hu | 1, 1);
}
model {
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N] mu = rep_vector(0.0, N);
    mu += Intercept + Xc * b;
    for (n in 1:N) {
      target += hurdle_neg_binomial_log_lpmf(Y[n] | mu[n], shape, hu);
    }
  }
  // priors including constants
  target += lprior;
}
generated quantities {
  // actual population-level intercept
  real b_Intercept = Intercept - dot_product(means_X, b);
}

