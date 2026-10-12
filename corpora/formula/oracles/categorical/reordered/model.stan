// generated with brms 2.23.2
functions {
}
data {
  int<lower=1> N;  // total number of observations
  int<lower=2> ncat;  // number of categories
  array[N] int Y;  // response variable
  int<lower=1> K_muzeta;  // number of population-level effects
  matrix[N, K_muzeta] X_muzeta;  // population-level design matrix
  int<lower=1> Kc_muzeta;  // number of population-level effects after centering
  int<lower=1> K_mualpha;  // number of population-level effects
  matrix[N, K_mualpha] X_mualpha;  // population-level design matrix
  int<lower=1> Kc_mualpha;  // number of population-level effects after centering
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
  matrix[N, Kc_muzeta] Xc_muzeta;  // centered version of X_muzeta without an intercept
  vector[Kc_muzeta] means_X_muzeta;  // column means of X_muzeta before centering
  matrix[N, Kc_mualpha] Xc_mualpha;  // centered version of X_mualpha without an intercept
  vector[Kc_mualpha] means_X_mualpha;  // column means of X_mualpha before centering
  for (i in 2:K_muzeta) {
    means_X_muzeta[i - 1] = mean(X_muzeta[, i]);
    Xc_muzeta[, i - 1] = X_muzeta[, i] - means_X_muzeta[i - 1];
  }
  for (i in 2:K_mualpha) {
    means_X_mualpha[i - 1] = mean(X_mualpha[, i]);
    Xc_mualpha[, i - 1] = X_mualpha[, i] - means_X_mualpha[i - 1];
  }
}
parameters {
  vector[Kc_muzeta] b_muzeta;  // regression coefficients
  real Intercept_muzeta;  // temporary intercept for centered predictors
  vector[Kc_mualpha] b_mualpha;  // regression coefficients
  real Intercept_mualpha;  // temporary intercept for centered predictors
}
transformed parameters {
  // prior contributions to the log posterior
  real lprior = 0;
  lprior += student_t_lpdf(Intercept_muzeta | 3, 0, 2.5);
  lprior += student_t_lpdf(Intercept_mualpha | 3, 0, 2.5);
}
model {
  // likelihood including constants
  if (!prior_only) {
    // joint regression coefficients over categories
    matrix[Kc_muzeta, ncat] b;
    // joint intercepts over categories
    vector[ncat] Intercept;
    b[, 1] = rep_vector(0, Kc_muzeta);
    b[, 2] = b_muzeta;
    b[, 3] = b_mualpha;
    Intercept = transpose([0, Intercept_muzeta, Intercept_mualpha]);
    target += categorical_logit_glm_lpmf(Y | Xc_muzeta, Intercept, b);
  }
  // priors including constants
  target += lprior;
}
generated quantities {
  // actual population-level intercept
  real b_muzeta_Intercept = Intercept_muzeta - dot_product(means_X_muzeta, b_muzeta);
  // actual population-level intercept
  real b_mualpha_Intercept = Intercept_mualpha - dot_product(means_X_mualpha, b_mualpha);
}

