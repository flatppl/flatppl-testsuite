# Rscript --vanilla generate.R OUTPUT_DIRECTORY
suppressPackageStartupMessages(library(brms))
stopifnot(as.character(getRversion()) == "4.5.3",
          as.character(packageVersion("brms")) == "2.23.2",
          as.character(packageVersion("mgcv")) == "1.9.4")
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
out <- args[[1]]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
json <- function(x) {
  if (is.matrix(x)) x <- lapply(seq_len(nrow(x)), function(i) unname(as.list(x[i, ])))
  if (is.list(x)) {
    values <- vapply(x, json, "")
    if (is.null(names(x))) return(paste0("[", paste(values, collapse = ","), "]"))
    keys <- vapply(names(x), encodeString, "", quote = '"')
    return(paste0("{", paste(paste0(keys, ":", values), collapse = ","), "}"))
  }
  if (length(x) != 1L) return(json(unname(as.list(x))))
  trimws(formatC(x, digits = 17, format = "g", decimal.mark = "."))
}
x <- c(-2.7,-2,-1.8,-1.1,-.6,-.1,.2,.9,1.2,1.8,2.1,2.8,3.4,4)
data <- data.frame(x = x, y = sin(x) + seq_along(x)/100)
formula <- bf(y ~ s(x, k = 6))
priors <- c(prior(normal(0, 2), class = "b"),
            prior(normal(0, 1), class = "Intercept"),
            prior(exponential(1), class = "sds"),
            prior(exponential(1), class = "sigma"))
writeLines(make_stancode(formula, data = data, prior = priors, normalize = TRUE),
           file.path(out, "model.stan"))
standata <- make_standata(formula, data = data, prior = priors)
standata <- standata[c("N", "Y", "Ks", "Xs", "nb_1", "knots_1", "Zs_1_1", "prior_only")]
standata$knots_1 <- unname(as.list(standata$knots_1))
writeLines(json(standata), file.path(out, "data.json"), useBytes = TRUE)

smooth <- brms:::smoothCon(mgcv::s(x, k = 6), data, absorb.cons = TRUE,
                          diagonal.penalty = TRUE)[[1]]
random <- mgcv::smooth2random(smooth, names(data), type = 2)
new_x <- c(min(x)-2, min(x), mean(x), max(x), max(x)+2)
predicted <- brms:::s2rPred(smooth, random, data.frame(x = new_x))
prediction <- list(x = new_x, fixed = unname(predicted$Xf),
                   random = unname(predicted$rand[[1]]),
                   mean = drop(.3 + .4 * predicted$Xf +
                               .7 * predicted$rand[[1]] %*% c(-.2,.1,.35,-.15)))
writeLines(json(prediction), file.path(out, "prediction.json"), useBytes = TRUE)
