// generated with brms 2.23.2
functions {
}
data {
  int<lower=1> N;  // total number of observations
  int<lower=2> ncat;  // number of categories
  array[N] int Y;  // response variable
  int<lower=1> K_mublue;  // number of population-level effects
  matrix[N, K_mublue] X_mublue;  // population-level design matrix
  int<lower=1> Kc_mublue;  // number of population-level effects after centering
  int<lower=1> K_mugreen;  // number of population-level effects
  matrix[N, K_mugreen] X_mugreen;  // population-level design matrix
  int<lower=1> Kc_mugreen;  // number of population-level effects after centering
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
  matrix[N, Kc_mublue] Xc_mublue;  // centered version of X_mublue without an intercept
  vector[Kc_mublue] means_X_mublue;  // column means of X_mublue before centering
  matrix[N, Kc_mugreen] Xc_mugreen;  // centered version of X_mugreen without an intercept
  vector[Kc_mugreen] means_X_mugreen;  // column means of X_mugreen before centering
  for (i in 2:K_mublue) {
    means_X_mublue[i - 1] = mean(X_mublue[, i]);
    Xc_mublue[, i - 1] = X_mublue[, i] - means_X_mublue[i - 1];
  }
  for (i in 2:K_mugreen) {
    means_X_mugreen[i - 1] = mean(X_mugreen[, i]);
    Xc_mugreen[, i - 1] = X_mugreen[, i] - means_X_mugreen[i - 1];
  }
}
parameters {
  vector[Kc_mublue] b_mublue;  // regression coefficients
  real Intercept_mublue;  // temporary intercept for centered predictors
  vector[Kc_mugreen] b_mugreen;  // regression coefficients
  real Intercept_mugreen;  // temporary intercept for centered predictors
}
transformed parameters {
  // prior contributions to the log posterior
  real lprior = 0;
  lprior += student_t_lpdf(Intercept_mublue | 3, 0, 2.5);
  lprior += student_t_lpdf(Intercept_mugreen | 3, 0, 2.5);
}
model {
  // likelihood including constants
  if (!prior_only) {
    // joint regression coefficients over categories
    matrix[Kc_mublue, ncat] b;
    // joint intercepts over categories
    vector[ncat] Intercept;
    b[, 1] = rep_vector(0, Kc_mublue);
    b[, 2] = b_mublue;
    b[, 3] = b_mugreen;
    Intercept = transpose([0, Intercept_mublue, Intercept_mugreen]);
    target += categorical_logit_glm_lpmf(Y | Xc_mublue, Intercept, b);
  }
  // priors including constants
  target += lprior;
}
generated quantities {
  // actual population-level intercept
  real b_mublue_Intercept = Intercept_mublue - dot_product(means_X_mublue, b_mublue);
  // actual population-level intercept
  real b_mugreen_Intercept = Intercept_mugreen - dot_product(means_X_mugreen, b_mugreen);
}

