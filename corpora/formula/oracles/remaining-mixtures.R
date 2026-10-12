encode_json <- function(x) {
  if (is.null(x)) return("null")
  if (is.list(x)) {
    values <- vapply(x, encode_json, "")
    if (is.null(names(x))) return(paste0("[", paste(values, collapse=","), "]"))
    keys <- vapply(names(x), encodeString, "", quote='"')
    return(paste0("{", paste(paste0(keys, ":", values), collapse=","), "}"))
  }
  if (length(x) != 1L) return(encode_json(unname(as.list(x))))
  if (is.character(x)) return(encodeString(x, quote='"'))
  if (is.logical(x)) return(if (x) "true" else "false")
  stopifnot(is.numeric(x), is.finite(x))
  trimws(formatC(x, digits=17, format="g", decimal.mark="."))
}

x <- seq_len(4) - 2.5
families <- c("zero_inflated_beta", "hurdle_poisson", "hurdle_negbinomial",
              "hurdle_gamma", "hurdle_lognormal")
response <- list(zero_inflated_beta=c(0,.2,.8,0), hurdle_poisson=c(0,2,4,0),
  hurdle_negbinomial=c(0,2,4,0), hurdle_gamma=c(0,.8,2.3,0),
  hurdle_lognormal=c(0,exp(-1),exp(2),0))
locations <- c(zero_inflated_beta=0, hurdle_poisson=-.8,
               hurdle_negbinomial=-.8, hurdle_gamma=-1.3, hurdle_lognormal=-1.7)

target <- function(family, p) {
  y <- response[[family]]
  positive <- y > 0
  eta <- p[[1]] + p[[2]] * x
  mass <- tail(p,1)
  q <- y[positive]
  mu <- exp(eta[positive])
  aux <- p[[3]]
  daux <- rep(0,length(q))
  if (family == "hurdle_poisson") {
    logp <- dpois(q,mu,log=TRUE)-log(-expm1(-mu))
    deta <- q-mu-mu/expm1(mu)
  } else if (family == "hurdle_negbinomial") {
    logp0 <- aux*(log(aux)-log(aux+mu))
    odds0 <- exp(logp0)/(-expm1(logp0))
    logp <- dnbinom(q,size=aux,mu=mu,log=TRUE)-log(-expm1(logp0))
    deta <- aux*(q-mu)/(aux+mu)-odds0*aux*mu/(aux+mu)
    daux <- digamma(q+aux)-digamma(aux)+log(aux)+1-log(aux+mu)-(aux+q)/(aux+mu) +
      odds0*(log(aux)-log(aux+mu)+1-aux/(aux+mu))
  } else if (family == "hurdle_gamma") {
    logp <- dgamma(q,shape=aux,rate=aux/mu,log=TRUE)
    deta <- aux*(q/mu-1)
    daux <- log(aux)+1-log(mu)-digamma(aux)+log(q)-q/mu
  } else if (family == "hurdle_lognormal") {
    delta <- log(q)-eta[positive]
    logp <- dlnorm(q,meanlog=eta[positive],sdlog=aux,log=TRUE)
    deta <- delta/aux^2
    daux <- -1/aux+delta^2/aux^3
  } else {
    mu <- plogis(eta[positive])
    alpha <- mu*aux
    beta <- (1-mu)*aux
    logp <- dbeta(q,alpha,beta,log=TRUE)
    deta <- aux*mu*(1-mu)*(log(q)-log1p(-q)-digamma(alpha)+digamma(beta))
    daux <- digamma(aux)-mu*digamma(alpha)-(1-mu)*digamma(beta) +
      mu*log(q)+(1-mu)*log1p(-q)
  }
  pointwise <- rep(log(mass),length(y))
  pointwise[positive] <- log1p(-mass)+logp
  mass_gradient <- sum(!positive)/mass-sum(positive)/(1-mass)
  gradient <- c(sum(deta),sum(deta*x[positive]))
  if (family != "hurdle_poisson") gradient <- c(gradient,sum(daux))
  gradient <- c(gradient,mass_gradient)
  forward <- sum(pointwise)
  forward_gradient <- gradient
  delta <- p[[1]]-locations[[family]]
  prior <- dt(delta/2.5,df=3,log=TRUE)-log(2.5)
  gradient[[1]] <- gradient[[1]]-4*delta/(3*2.5^2+delta^2)
  if (family == "hurdle_negbinomial") {
    prior <- prior+.4*log(.3)-lgamma(.4)-1.4*log(aux)-.3/aux
    gradient[[3]] <- gradient[[3]]-1.4/aux+.3/aux^2
  } else if (family == "hurdle_lognormal") {
    prior <- prior+dt(aux/2.5,df=3,log=TRUE)-log(2.5)+log(2)
    gradient[[3]] <- gradient[[3]]-4*aux/(3*2.5^2+aux^2)
  } else if (family != "hurdle_poisson") {
    prior <- prior+dgamma(aux,shape=.01,rate=.01,log=TRUE)
    gradient[[3]] <- gradient[[3]]-.99/aux-.01
  }
  list(point=unname(p),posterior=forward+prior,gradient=unname(gradient),
       forward=forward,forward_gradient=unname(forward_gradient),pointwise=pointwise)
}

cases <- setNames(lapply(families,function(family) {
  points <- if (family == "hurdle_poisson") {
    list(c(.4,.3,.3),c(-.7,-.2,.65))
  } else if (family == "hurdle_lognormal") {
    list(c(.4,.3,1.2,.3),c(-.7,-.2,.75,.65))
  } else list(c(.4,.3,2.3,.3),c(-.7,-.2,.8,.65))
  lapply(points,function(p) target(family,p))
}),families)
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
for (family in names(cases)) {
  directory <- file.path(args[[1]], paste0("family_", family))
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  writeLines(encode_json(cases[[family]]), file.path(directory, "base-r.json"))
}
