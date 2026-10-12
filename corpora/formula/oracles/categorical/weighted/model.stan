// generated with brms 2.23.2
functions {
}
data {
  int<lower=1> N;  // total number of observations
  int<lower=2> ncat;  // number of categories
  array[N] int Y;  // response variable
  vector<lower=0>[N] weights;  // model weights
  int<lower=1> K_mublue;  // number of population-level effects
  matrix[N, K_mublue] X_mublue;  // population-level design matrix
  int<lower=1> K_mugreen;  // number of population-level effects
  matrix[N, K_mugreen] X_mugreen;  // population-level design matrix
  int<lower=1> Kc_mugreen;  // number of population-level effects after centering
  vector[N] offsets_mugreen;
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
  matrix[N, Kc_mugreen] Xc_mugreen;  // centered version of X_mugreen without an intercept
  vector[Kc_mugreen] means_X_mugreen;  // column means of X_mugreen before centering
  for (i in 2:K_mugreen) {
    means_X_mugreen[i - 1] = mean(X_mugreen[, i]);
    Xc_mugreen[, i - 1] = X_mugreen[, i] - means_X_mugreen[i - 1];
  }
}
parameters {
  vector[K_mublue] b_mublue;  // regression coefficients
  vector[Kc_mugreen] b_mugreen;  // regression coefficients
  real Intercept_mugreen;  // temporary intercept for centered predictors
}
transformed parameters {
  // prior contributions to the log posterior
  real lprior = 0;
  lprior += normal_lpdf(b_mublue | 0, 1.1);
  lprior += normal_lpdf(b_mugreen | 0, 0.7);
  lprior += normal_lpdf(Intercept_mugreen | 0.2, 1.3);
}
model {
  // likelihood including constants
  if (!prior_only) {
    // initialize linear predictor term
    vector[N] mublue = rep_vector(0.0, N);
    // initialize linear predictor term
    vector[N] mugreen = rep_vector(0.0, N);
    // linear predictor matrix
    array[N] vector[ncat] mu;
    mublue += X_mublue * b_mublue;
    mugreen += Intercept_mugreen + Xc_mugreen * b_mugreen + offsets_mugreen;
    for (n in 1:N) {
      mu[n] = transpose([0, mublue[n], mugreen[n]]);
    }
    for (n in 1:N) {
      target += weights[n] * (categorical_logit_lpmf(Y[n] | mu[n]));
    }
  }
  // priors including constants
  target += lprior;
}
generated quantities {
  // actual population-level intercept
  real b_mugreen_Intercept = Intercept_mugreen - dot_product(means_X_mugreen, b_mugreen);
}

