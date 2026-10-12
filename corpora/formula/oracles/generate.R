# Generate the corpus oracles with R 4.5.3 and pinned brms 2.23.2.
# Usage: Rscript --vanilla generate.R OUTPUT_DIRECTORY
suppressPackageStartupMessages(library(brms))
stopifnot(as.character(getRversion()) == "4.5.3",
          as.character(packageVersion("brms")) == "2.23.2")
Sys.setlocale("LC_COLLATE", "C")
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
out <- args[[1]]
dir.create(out, recursive = TRUE, showWarnings = FALSE)

encode_json <- function(x) {
  if (is.null(x)) return("null")
  if (is.factor(x)) x <- as.character(x)
  if (is.matrix(x)) x <- lapply(seq_len(nrow(x)), function(i) unname(as.list(x[i, ])))
  if (is.list(x)) {
    if (!length(x)) return(if (is.null(names(x))) "[]" else "{}")
    values <- vapply(x, encode_json, "")
    if (is.null(names(x))) return(paste0("[", paste(values, collapse = ","), "]"))
    keys <- vapply(names(x), encodeString, "", quote = '"')
    return(paste0("{", paste(paste0(keys, ":", values), collapse = ","), "}"))
  }
  if (length(x) != 1L) return(encode_json(unname(as.list(x))))
  if (is.na(x)) return("null")
  if (is.character(x)) return(encodeString(x, quote = '"'))
  if (is.logical(x)) return(if (x) "true" else "false")
  if (is.numeric(x)) return(if (is.finite(x)) trimws(formatC(x, digits = 17, format = "g", decimal.mark = ".")) else "null")
  stop("unsupported JSON value")
}
json <- function(value, path) {
  writeLines(encode_json(value), path, useBytes = TRUE)
}
data <- data.frame(y = c(2, 4, 8, 7), x = 1:4, z = c(2, 3, 5, 7),
                   g = factor(c("A", "B", "A", "B"), levels = c("B", "A")),
                   h = factor(c("1", "2", "2", "1")),
                   f = factor(c("b", "a", "b", "a"), levels = c("b", "a")),
                   trials = c(3L, 5L, 8L, 9L))

capture <- function(name, formula, family, data = data, prior = NULL) {
  args <- list(formula = formula, data = data, family = family)
  prior_rows <- do.call(get_prior, args)
  if (!is.null(prior)) args$prior <- prior
  standata <- do.call(make_standata, args)
  code <- do.call(make_stancode, c(args, list(normalize = TRUE)))
  directory <- file.path(out, name)
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  if (name == "family_gaussian_softit") {
    # brms 2.23.2 emits invalid vector/vector division in both vector overloads.
    writeLines(code, file.path(directory, "brms-generated.stan"))
    code <- sub("vector softit(vector p) {\n     return log(expm1(-p /",
                "vector softit(vector p) {\n     return log(expm1(-p ./", code, fixed = TRUE)
    code <- sub("vector inv_softit(vector y) {\n     return log1p_exp(y) /",
                "vector inv_softit(vector y) {\n     return log1p_exp(y) ./", code, fixed = TRUE)
  }
  writeLines(code, file.path(directory, "model.stan"))
  json(standata, file.path(directory, "data.json"))
  list(name = name, formula = paste(deparse(formula), collapse = " "),
       family = family$family, link = family$link,
       data = data, default_priors = as.data.frame(prior_rows),
       supplied_priors = if (is.null(prior)) NULL else as.data.frame(prior),
       factor_levels = lapply(data[vapply(data, is.factor, TRUE)], levels),
       matrix_columns = lapply(standata[vapply(standata, is.matrix, TRUE)], colnames),
       standata = standata)
}

families <- list(gaussian = gaussian(), student = student(),
                 bernoulli = bernoulli(), binomial = binomial(),
                 poisson = poisson(), negbinomial = negbinomial(),
                 Gamma = brmsfamily("Gamma"), exponential = exponential(),
                 lognormal = brmsfamily("lognormal"), geometric = brmsfamily("geometric"),
                 weibull = brmsfamily("weibull"), beta = brmsfamily("beta"),
                 von_mises = brmsfamily("von_mises"))
responses <- list(lognormal = exp(c(-3, -1, 2, 3)), geometric = c(0L, 4L, 8L, 7L),
                  beta = c(.1, .4, .8, .7), von_mises = c(-2, -.4, .8, 2))
for (name in names(families)) {
  d <- data
  if (name == "bernoulli") d$y <- c(0L, 1L, 1L, 0L)
  if (name == "binomial") d$y <- c(1L, 2L, 4L, 3L)
  if (name %in% names(responses)) d$y <- responses[[name]]
  f <- if (name == "binomial") y | trials(trials) ~ x else y ~ x
  directory <- paste0("family_", tolower(name))
  captured <- capture(directory, f, families[[name]], d)
  json(captured, file.path(out, directory, "brms.json"))
}

priors <- c(set_prior("normal(0, 2)", class = "b"),
            set_prior("normal(1, 3)", class = "Intercept"),
            set_prior("normal(0, 1)", class = "sigma"))
expressions <- capture("expressions_offset",
  y ~ I(x^2) + log(z) + offset(log(z)), gaussian(), data, priors)
json(expressions, file.path(out, "expressions_offset", "brms.json"))

groups <- list(slopes_correlated = y ~ x + (1 + x | g),
               slopes_independent = y ~ x + (1 + x || g))
for (name in names(groups)) {
  captured <- capture(name, groups[[name]], gaussian(), data,
    c(priors, set_prior("normal(0, 1)", class = "sd")))
  json(captured, file.path(out, name, "brms.json"))
}

log_data <- data
log_data$y <- c(-2, 4, 8, 7)
offset <- suppressWarnings(capture("default_log_offset",
  y ~ x + offset(log(z)), gaussian("log"), log_data))
json(offset, file.path(out, "default_log_offset", "brms.json"))

group_priors <- c(priors,
  set_prior("normal(0, 2)", class = "sd"),
  set_prior("cauchy(0, 1)", class = "sd", group = "g"),
  set_prior("student_t(4, 1, 3)", class = "sd", group = "g", coef = "x"),
  set_prior("lkj(2)", class = "cor"),
  set_prior("lkj(3)", class = "cor", group = "g"))
selected <- capture("group_priors", y ~ x + (1 + x | g) + (1 + x | h),
  gaussian(), data, group_priors)
json(selected, file.path(out, "group_priors", "brms.json"))

softit <- capture("family_gaussian_softit", y ~ x, brmsfamily("gaussian", link = "softit"), data)
json(softit, file.path(out, "family_gaussian_softit", "brms.json"))
dependent <- capture("dependent_locscale_prior", y ~ x, gaussian(), data,
  c(set_prior("normal(1, 3)", class = "Intercept"),
    set_prior("cauchy(Intercept, 1.7)", class = "sigma")))
json(dependent, file.path(out, "dependent_locscale_prior", "brms.json"))

ordinal_data <- data.frame(y = ordered(c("medium", "low", "high", "medium"),
  levels = c("low", "medium", "high")), x = c(1, 2, 4, 5))
ordinal <- capture("cumulative_logit", y ~ x, cumulative("logit"), ordinal_data,
  c(set_prior("normal(0, 2)", class = "Intercept"),
    set_prior("normal(0, 1.5)", class = "b")))
json(ordinal, file.path(out, "cumulative_logit", "brms.json"))
ordinal_default <- capture("cumulative_logit_default", y ~ x,
  cumulative("logit"), ordinal_data)
json(ordinal_default, file.path(out, "cumulative_logit_default", "brms.json"))

weighted_data <- data.frame(y = c(1, 2, 4, 7), x = c(-1, 0, 1, 2),
                            w = c(0, 0.5, 2, 3), n = c(2L, 3L, 5L, 8L))
weighted_priors <- c(set_prior("normal(0, 2)", class = "b"),
                     set_prior("normal(0, 3)", class = "Intercept"))
for (family_name in c("gaussian", "binomial")) {
  f <- if (family_name == "binomial") y | trials(n) + weights(w) ~ x else y | weights(w) ~ x
  prior <- weighted_priors
  if (family_name == "gaussian") prior <- c(prior, set_prior("exponential(1.2)", class = "sigma"))
  directory <- paste0("weighted_", family_name)
  captured <- capture(directory, f, brmsfamily(family_name), weighted_data, prior)
  json(captured, file.path(out, directory, "brms.json"))
}

se_data <- data.frame(y = c(1, 2, 4), x = c(0, 1, 3), s = c(.3, .6, .9))
for (family in c("gaussian", "student")) {
  for (residual in c(FALSE, TRUE)) {
    name <- paste0("se_", family, if (residual) "_residual" else "")
    formula <- as.formula(paste0("y | se(s, sigma=", residual, ") ~ x"))
    priors <- c(set_prior("normal(0, 2)", class = "Intercept"),
                set_prior("normal(0, 1)", class = "b"))
    if (residual) priors <- c(priors, set_prior("normal(0, 1)", class = "sigma"))
    if (family == "student") priors <- c(priors, set_prior("constant(5)", class = "nu"))
    law <- if (family == "student") student() else gaussian()
    captured <- capture(name, formula, law, se_data, priors)
    json(captured, file.path(out, name, "brms.json"))
  }
}
