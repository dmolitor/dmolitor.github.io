# Helper functions
projection_matrix <- function(X) {
  QRmat <- qr(X)
  Q <- qr.Q(QRmat)
  tcrossprod(Q)
}

MPxM <- function(M, X) {
  QRmat <- qr(X)
  Q <- qr.Q(QRmat)
  QtM <- crossprod(Q, M)
  crossprod(QtM)
}

MP1M <- function(M) {
  n <- nrow(M)
  if (is.null(n)) {
    M <- matrix(M, ncol = 1)
  }
  Mbar <- colSums(M)
  tcrossprod(Mbar) / n
}

# Sequential t-test: log Bayes factor for the univariate case
log_bf <- function(t2, nu, phi, z2) {
  r <- phi / (phi + z2)
  0.5 * log(r) + 0.5 * (nu + 1) * (log(1 + t2 / nu) - log(1 + r * t2 / nu))
}

# Multivariate sequential F-test: log Bayes factor
log_bf_multivariate <- function(delta, n, p, d, Phi, ZtZ, s2) {
  if (d > 1) {
    normalizing_constant <- 0.5 * log(det(Phi)) - 0.5 * log(det(Phi + ZtZ))
  } else {
    normalizing_constant <- 0.5 * log(Phi) - 0.5 * log(Phi + ZtZ)
  }
  sol <- if (d > 1) {
    solve(ZtZ + Phi, ZtZ)
  } else {
    ZtZ / (ZtZ + Phi)
  }
  normalizing_constant +
    0.5 * (n - p) * (
      log(1 + t(delta) %*% ZtZ %*% delta / (s2 * (n - p - d))) -
      log(1 + t(delta) %*% (ZtZ - ZtZ %*% sol) %*% delta / (s2 * (n - p - d)))
    )
}

# Sequential t-test p-value using the univariate log_bf
sequential_t_p_value <- function(model, treat_name, phi = 1, index = 2) {
  sm <- summary(model)
  coefs <- sm$coefficients
  # Assume the parameter of interest (δ) is the last coefficient
  delta_hat <- coefs[treat_name, "Estimate"]
  se_delta  <- coefs[treat_name, "Std. Error"]
  t_val     <- delta_hat / se_delta
  df        <- model$df.residual
  sigma2    <- sm$sigma^2
  # Compute ||Z̃||^2 = sigma2 / se_delta^2, which is the reciprocal
  # of the lower-right element of (W'W)^{-1}.
  z2 <- sigma2 / (se_delta^2)
  bf_log <- log_bf(t_val^2, df, phi, z2)
  pval <- min(1, exp(-bf_log))
  list(
    estimate = delta_hat,
    p_value = pval
  )
}

# Sequential t-test confidence sequence
sequential_t_cs <- function(model, treat_name, alpha = 0.05, phi = 1, index = 2) {
  # Extract basic statistics from the lm object
  sm <- summary(model)
  coefs <- sm$coefficients
  delta_hat <- coefs[treat_name, "Estimate"]
  se_delta  <- coefs[treat_name, "Std. Error"]
  t_val     <- delta_hat / se_delta
  # Degrees of freedom (nu) is taken from the model's residual degrees of freedom
  nu <- model$df.residual
  # sigma^2 is the residual variance
  sigma2 <- sm$sigma^2
  # Compute z2 = sigma2/(se_delta^2), which is the reciprocal of the lower-right entry
  # of (W'W)^{-1} in our notation.
  z2 <- sigma2 / (se_delta^2)
  # For the sequential t-test, the tuning parameter phi is a scalar.
  # Compute r = phi/(phi + z2)
  r <- phi / (phi + z2)
  # Now, following Corollary 3.8, the confidence sequence radius is computed as:
  # r_n = se_delta * sqrt( nu * {1 - (r * alpha^2)^(1/(nu+1))} / {max( (r * alpha^2)^(1/(nu+1)) - r, eps) } )
  eps <- 1e-8  # a small number to avoid division by zero
  term <- (r * alpha^2)^(1 / (nu + 1))
  numerator <- 1 - term
  denominator <- max(term - r, eps)
  radii <- se_delta * sqrt(nu * (numerator / denominator))
  lower <- delta_hat - radii
  upper <- delta_hat + radii
  # Return a list with both the sequential test statistic details and the CS bounds.
  list(
    estimate = delta_hat,
    cs_lower = lower,
    cs_upper = upper,
    # Also include the sequential p-value (which is 1/B_n) for reference.
    p_value = min(1, exp(-1 * log_bf(t_val^2, nu, phi, z2)))
  )
}

# Multivariate sequential F-test p-value
# Here we assume that the first column of the model matrix is nuisance,
# and the remaining columns correspond to the parameters of interest.
sequential_f_p_value <- function(model, phi = 1, s2 = NULL) {
  W <- model.matrix(model)
  p <- 1                   # Number of nuisance parameters (assumed to be 1)
  n <- nrow(W)
  d <- ncol(W) - 1         # Remaining columns correspond to δ (could be multivariate)
  Z <- W[, 2:(d + p), drop = FALSE]
  # Reconstruct Y from fitted values and residuals:
  Y <- model$fitted.values + model$residuals
  delta <- model$coefficients[2:(d + p)]
  # Compute  ̃ Z' ̃ Z = MPxM(Z,W) - MP1M(Z)
  ZtZ <- MPxM(Z, W) - MP1M(Z)
  # Use s2 from argument if provided; otherwise, use residual variance from lm
  if (is.null(s2)) {
    s2 <- summary(model)$sigma^2
  }
  Phi <- diag(d) * phi
  pval <- min(1, exp(-log_bf_multivariate(delta, n, p, d, Phi, ZtZ, s2)))
  pval
}

sequential_f_cs <- function(model, alpha = 0.05, phi = 1) {
  # Get the standard summary from lm
  mod <- summary(model)
  # Determine dimensions from the f-statistic:
  # fstatistic[2] is (p - 1) so p = fstatistic[2] + 1
  # n = df.residual + p, which equals fstatistic[3] + p.
  p <- mod$fstatistic[2] + 1  
  n <- mod$fstatistic[3] + p
  # Extract coefficients info
  coefs <- mod$coefficients
  estimates <- coefs[, "Estimate"]
  stderrs   <- coefs[, "Std. Error"]
  tstats    <- coefs[, "t value"]
  regular_p <- coefs[, "Pr(>|t|)"]
  tstats2   <- tstats^2
  # Residual standard error and its variance
  s <- mod$sigma
  s2 <- s^2
  # Compute z2 = (s / se)^2; this relates to the (W'W)⁻¹ element.
  z2 <- (s / stderrs)^2
  # Degrees of freedom for the t statistic
  nu <- n - p - 1
  # Compute sequential p-values for each coefficient using log_bf (univariate)
  spvalues <- pmin(1, exp(-1 * log_bf(tstats2, nu, phi, z2)))
  # Compute confidence sequence radii:
  # r is defined as phi/(phi + z2)
  r <- phi / (phi + z2)
  # To avoid division by zero, we use pmax with a small number
  radii <- stderrs * sqrt(nu * ((1 - (r * alpha^2)^(1 / (nu + 1))) /
              pmax((r * alpha^2)^(1 / (nu + 1)) - r, 1e-8)))
  # Confidence sequence lower and upper bounds
  lowers <- estimates - radii
  uppers <- estimates + radii
  # Compute the multivariate sequential F-test p-value for the nuisance parameters.
  # (This is provided as an overall model statistic.)
  f_seq_p <- sequential_f_p_value(model, phi, s2)
  # Construct a data frame with the results
  result_df <- data.frame(
    covariate = rownames(coefs),
    estimate = estimates,
    std_error = stderrs,
    t_stat = tstats,
    f_p_value = f_seq_p, # Multivariate sequential F-test p-value
    p_value = spvalues,
    cs_lower = lowers,
    cs_upper = uppers,
    stringsAsFactors = FALSE
  )
  return(result_df)
}

ate_cs_asymp <- function(model, alpha = 0.05, lambda = 10, treat_name = NULL, rho = NULL) {
  mod_sum <- summary(model)
  coefs <- mod_sum$coefficients
  # Identify the treatment effect coefficient:
  # If treat_name is provided, use that; otherwise assume the first non-intercept coefficient.
  if (!is.null(treat_name)) {
    if (!(treat_name %in% rownames(coefs))) {
      stop("Treatment variable not found in model coefficients.")
    }
    tau_hat <- coefs[treat_name, "Estimate"]
  } else {
    # Assumes the treatment effect is the first coefficient after the intercept.
    tau_hat <- coefs[2, "Estimate"]
    treat_name <- rownames(coefs)[2]
  }
  # Extract the residual standard error as a consistent estimator of σ.
  sigma_hat <- mod_sum$sigma
  # Determine the sample size
  n <- nrow(model$model)
  # Compute the treatment assignment probability, ρ.
  # If not provided, try to extract the treatment variable from the model data.
  if (is.null(rho)) {
    if (treat_name %in% names(model$model)) {
      T_vec <- model$model[[treat_name]]
      # Assume T is a 0/1 variable.
      rho <- mean(T_vec)
    } else {
      stop("Treatment probability (rho) not provided and treatment variable not found in model data.")
    }
  }
  # Compute the half-width r_n as in Equation (20):
  # r_n = sigma_hat / sqrt(n * ρ*(1-ρ)) * sqrt(((λ+n)/n) * log((λ+n)/(λ * α^2)))
  r_n <- (sigma_hat / sqrt(n * rho * (1 - rho))) * 
         sqrt(((lambda + n) / n) * log((lambda + n) / (lambda * alpha^2)))
  # Construct the confidence sequence for the ATE
  lower <- tau_hat - r_n
  upper <- tau_hat + r_n
  # Return results as a list (can be converted to a data.frame if needed)
  result <- list(
    estimate = tau_hat,
    sigma_hat = sigma_hat,
    cs_lower = lower,
    cs_upper = upper
  )
  return(result)
}