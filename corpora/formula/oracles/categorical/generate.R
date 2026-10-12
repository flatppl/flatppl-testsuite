# Usage: Rscript --vanilla generate.R OUTPUT_DIRECTORY
suppressPackageStartupMessages(library(brms))
stopifnot(as.character(getRversion()) == "4.5.3",
          as.character(packageVersion("brms")) == "2.23.2")
Sys.setlocale("LC_COLLATE", "C")
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
out <- args[[1]]

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
json <- function(value, path) writeLines(encode_json(value), path, useBytes = TRUE)
data <- data.frame(
  y = factor(c("red", "blue", "green", "red", "green", "blue"),
             levels = c("unused", "red", "blue", "green")),
  x = 1:6, z = c(0, 1, 0, 2, 1, 0), w = c(1, .5, 2, 0, 1.25, 1))
formulas <- list(
  default = bf(y ~ x),
  weighted = bf(y | weights(w) ~ x, mublue ~ 0 + z, mugreen ~ x + offset(z)),
  reordered = y ~ x)
priors <- c(prior(normal(0, 1.1), class = b, dpar = mublue),
            prior(normal(0, .7), class = b, dpar = mugreen),
            prior(normal(.2, 1.3), class = Intercept, dpar = mugreen))
for (name in names(formulas)) {
  d <- data
  if (name == "reordered") levels(d$y) <- c("unused", "red", "zeta", "alpha")
  p <- if (name == "weighted") priors else NULL
  f <- formulas[[name]]
  directory <- file.path(out, name)
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  writeLines(make_stancode(f, d, family = categorical(), prior = p, normalize = TRUE),
             file.path(directory, "model.stan"))
  json(make_standata(f, d, family = categorical(), prior = p),
       file.path(directory, "data.json"))
  json(list(family = "categorical", formula = paste(deparse(f), collapse = " "),
            data = d, default_priors = as.data.frame(get_prior(f, d, family = categorical())),
            supplied_priors = if (is.null(p)) NULL else as.data.frame(p),
            response_levels = levels(d$y)), file.path(directory, "brms.json"))
}
