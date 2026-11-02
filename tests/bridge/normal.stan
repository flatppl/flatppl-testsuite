// Two independent conjugate Normal means: the posterior is Normal in closed form.
data {
  int<lower=0> N;
  vector[N] y1;
  vector[N] y2;
}
parameters {
  vector[2] mu;
}
model {
  mu ~ normal(1.5, 2);
  y1 ~ normal(mu[1], 1);
  y2 ~ normal(mu[2], 1);
}
