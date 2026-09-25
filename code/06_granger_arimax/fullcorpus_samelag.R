#!/usr/bin/env Rscript
#
# full-corpus granger causality and ARIMAX analysis


E    <- "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/embeddings"
DATA <- "/home/local/PSYCH-ADS/avamadesousa/YES_lab/RelationalNormsNLP/RelationalNormsNLP_clean/data"
OUT  <- file.path(E, "SamelagGrangerArimax", "full_corpus")
if (!dir.exists(OUT)) dir.create(OUT, recursive = TRUE)
DOMS    <- c("care", "hierarchy", "transaction", "mating")
CORPORA <- c("US", "Chinese")
FDR_FAMILIES <- list(c("indi", "coll"), c("tight", "loose"))
DIRECTIONS <- c("culture_to_domain", "domain_to_culture")

CHINA_PREVALENCE_2019 <- read.csv(file.path(DATA, "china_cultural_values_and_gdp_2019.csv"))

# China's score is read directly from FullCorpusScores/ (03_scoring/'s own output); its
# indi/coll/tight/loose/gdp come from CHINA_PREVALENCE_2019, not from a current-vintage Ngram
# panel (see the header comment above and 08_granger_arimax/README.md). US reads its score and
# covariates together from FullCorpusTimeSeriesCovariates/, built by 04_cultural_covariates/.
load_cell <- function(corpus, dom) {
  if (corpus == "Chinese") {
    raw <- read.csv(file.path(E, "FullCorpusScores", "Chinese", paste0("avg__", dom, ".csv")))
    raw <- raw[order(raw$year), ]
    df <- data.frame(year = raw$year, score = raw$mean)
    df <- merge(df, CHINA_PREVALENCE_2019, by = "year")
  } else {
    df <- read.csv(file.path(E, "FullCorpusTimeSeriesCovariates", corpus, paste0("avg__", dom, ".csv")))
  }
  df <- df[order(df$year), ]
  df$gdp_z <- as.vector(scale(log(df$gdp)))
  for (p in c("indi", "coll", "tight", "loose")) df[[paste0(p, "_z")]] <- as.vector(scale(df[[p]]))
  df
}

perform_granger_test <- function(cause, effect) {
  df <- data.frame(y = effect, x = cause)
  df <- df[complete.cases(df), ]

  d_y <- ndiffs(df$y, test = "adf")
  d_x <- ndiffs(df$x, test = "adf")
  diff_order <- max(d_y, d_x)
  if (diff_order > 0) {
    df <- data.frame(y = diff(df$y, differences = diff_order),
                      x = diff(df$x, differences = diff_order))
  }

  n_obs <- nrow(df)
  lag_max <- max(1, floor(n_obs / 10))
  var_result <- VARselect(df, lag.max = lag_max, type = "both")
  optimal_lag <- var_result$selection["AIC(n)"]
  if (optimal_lag < 1) optimal_lag <- 1

  var_model <- VAR(df, p = optimal_lag)
  gt <- causality(var_model, cause = "x")
  # Peak lag: the largest |z| among the cause's lag coefficients in Granger's own VAR
  # (equation for the effect variable).
  zt <- coef(var_model)$y[paste0("x.l", seq_len(unname(optimal_lag))), "t value"]
  pk <- which.max(abs(zt))
  list(diff_order = diff_order, n_obs = n_obs, lag_max = lag_max,
       best_lag = unname(optimal_lag), p_value = gt$Granger$p.value,
       peak_lag = unname(pk), peak_z = unname(zt[pk]))
}

granger_for_cell <- function(df, corpus, domain) {
  rows <- list(); k <- 1
  for (p in c("indi", "coll", "tight", "loose")) {
    x <- ts(df[[paste0(p, "_z")]]); y <- ts(df$score)
    fwd <- perform_granger_test(x, y)
    rev <- perform_granger_test(y, x)
    rows[[k]] <- data.frame(corpus, domain, predictor = p, direction = "culture_to_domain",
      family = if (p %in% c("indi","coll")) "indi_coll" else "tight_loose",
      n_obs = fwd$n_obs, diff_order = fwd$diff_order, best_lag = fwd$best_lag,
      p_value = fwd$p_value, granger_peak_lag = fwd$peak_lag, granger_peak_z = fwd$peak_z,
      stringsAsFactors = FALSE); k <- k + 1
    rows[[k]] <- data.frame(corpus, domain, predictor = p, direction = "domain_to_culture",
      family = if (p %in% c("indi","coll")) "indi_coll" else "tight_loose",
      n_obs = rev$n_obs, diff_order = rev$diff_order, best_lag = rev$best_lag,
      p_value = rev$p_value, granger_peak_lag = rev$peak_lag, granger_peak_z = rev$peak_z,
      stringsAsFactors = FALSE); k <- k + 1
  }
  do.call(rbind, rows)
}

build_lagged_xreg <- function(x, n_lags) {
  lagmat <- embed(x, n_lags + 1)
  xreg <- lagmat[, 2:(n_lags + 1), drop = FALSE]
  colnames(xreg) <- paste0("lag", seq_len(n_lags))
  list(xreg = xreg, valid_idx = (n_lags + 1):length(x))
}

perform_arimax <- function(y, xreg_culture, gdp_z = NULL) {
  xreg <- xreg_culture
  if (!is.null(gdp_z)) xreg <- cbind(xreg_culture, gdp = gdp_z)
  lag_names <- colnames(xreg_culture)

  fit <- tryCatch(
    auto.arima(ts(y), xreg = xreg, d = 0, ic = "aicc",
               stepwise = FALSE, approximation = FALSE),
    error = function(e) e)
  if (inherits(fit, "error")) {
    return(list(order = NA, wald_p = NA, lb_p = NA, error = conditionMessage(fit), lag_coefs = NULL))
  }

  b <- coef(fit); V <- vcov(fit)
  ok_names <- lag_names[lag_names %in% names(b)]
  if (length(ok_names) < length(lag_names)) {
    return(list(order = sprintf("(%d,0,%d)", fit$arma[1], fit$arma[2]),
                wald_p = NA, lb_p = NA, error = "lag coefficient(s) not retained", lag_coefs = NULL))
  }
  Vsub <- V[ok_names, ok_names, drop = FALSE]
  bsub <- b[ok_names]
  Vinv <- tryCatch(solve(Vsub), error = function(e) NULL)
  if (is.null(Vinv)) { wald_stat <- NA; wald_p <- NA } else {
    wald_stat <- as.numeric(t(bsub) %*% Vinv %*% bsub)
    wald_p <- pchisq(wald_stat, df = length(ok_names), lower.tail = FALSE)
  }
  lb <- tryCatch(checkresiduals(fit, plot = FALSE), error = function(e) NULL)

  # Per-lag coefficient/SE/z/p -- the joint Wald test above has NO single sign
  # (H0 is "all lags = 0" simultaneously; individual lags could in principle
  # even have opposite signs), but each individual lag coefficient IS signed
  # and individually z-testable. Descriptive, not a re-test of the cell's
  # overall significance (that's still the joint Wald p above).
  se_all <- sqrt(diag(V))
  se <- se_all[ok_names]
  z_vals <- bsub / se
  p_vals <- 2 * pnorm(abs(z_vals), lower.tail = FALSE)

  # Standardized coefficient -- NOT redundant with the "_z" predictor columns
  # in load_cell(). Those only standardize the CULTURE variable; which side
  # (response y vs. lagged predictor xreg) the culture variable occupies
  # SWAPS by direction (arimax_for_row(): culture_to_domain -> response is
  # raw "score", predictor is culture_z; domain_to_culture -> reversed). So
  # a raw `coefficient` is in "raw-score units per z-unit" for one direction
  # and "z-units per raw-score unit" for the other -- NOT comparable across
  # directions (raw score has a tiny natural SD, so domain_to_culture
  # coefficients come out inflated by ~1/SD(score), e.g. coefficient=13.6
  # for a real, modest effect). Post-hoc standardization (coefficient *
  # SD(x_actual)/SD(y_actual), the standard OLS raw-to-standardized-beta
  # conversion) on the ACTUAL differenced/aligned y and xreg columns that
  # were actually fit -- not re-derived elsewhere, so it can't drift from
  # what the model actually saw -- makes every cell directly comparable
  # regardless of direction.
  sd_y <- sd(y)
  sd_x <- apply(xreg_culture[, ok_names, drop = FALSE], 2, sd)
  coefficient_std <- unname(bsub) * unname(sd_x) / sd_y
  se_std <- unname(se) * unname(sd_x) / sd_y
  lag_coefs <- data.frame(lag = seq_along(ok_names), coefficient = unname(bsub),
    se = unname(se), z = unname(z_vals), p = unname(p_vals),
    coefficient_std = coefficient_std, se_std = se_std, stringsAsFactors = FALSE)

  # Summed lag coefficient (cumulative effect of a sustained unit shift in x on y) with its SE
  # from the joint covariance: sqrt(1' V 1). Standardized with the mean lag-column SD so its z
  # equals the raw sum's z. Unlike the peak lag, no lag is selected, so its p is a valid test.
  ones <- rep(1, length(ok_names))
  sum_coef <- sum(bsub)
  sum_se <- sqrt(as.numeric(t(ones) %*% Vsub %*% ones))
  sum_z <- sum_coef / sum_se
  sum_stats <- list(coef = unname(sum_coef), se = sum_se, z = unname(sum_z),
                    p = unname(2 * pnorm(abs(sum_z), lower.tail = FALSE)),
                    coef_std = unname(sum_coef) * mean(sd_x) / sd_y, se_std = sum_se * mean(sd_x) / sd_y)

  list(order = sprintf("(%d,0,%d)", fit$arma[1], fit$arma[2]), aicc = fit$aicc,
       n_obs = length(y), wald_p = wald_p,
       lb_p = if (is.null(lb)) NA else unname(lb$p.value), error = NA, lag_coefs = lag_coefs,
       sum_stats = sum_stats)
}

# Per-lag sign pattern of the multi-lag fit (transparency only).
sign_summary <- function(lag_coefs) {
  if (is.null(lag_coefs)) return(list(pattern = NA_character_, n_pos = NA_integer_, n_neg = NA_integer_))
  list(pattern = paste0(ifelse(lag_coefs$coefficient >= 0, "+", "-"), collapse = ""),
       n_pos = sum(lag_coefs$coefficient > 0), n_neg = sum(lag_coefs$coefficient < 0))
}

# Direction and size of a cell: coefficient at the PEAK lag (largest |z| in Granger's VAR),
# from an ARIMAX refit containing ONLY that lag, so the other lags are not held constant.
single_cols <- function(res, prefix, k) {
  lc <- res$lag_coefs
  na <- is.null(lc)
  out <- data.frame(lag = k,
    coef = if (na) NA_real_ else lc$coefficient,
    coef_std = if (na) NA_real_ else lc$coefficient_std,
    se_std = if (na) NA_real_ else lc$se_std,
    z = if (na) NA_real_ else lc$z,
    p = if (na) NA_real_ else lc$p,
    sign = if (na) NA_character_ else ifelse(lc$coefficient >= 0, "+", "-"),
    stringsAsFactors = FALSE)
  names(out) <- paste0(prefix, "_peak_", names(out))
  out
}

# Summed-lag version of a cell (multi-lag fit): columns {prefix}_sum_*.
sum_cols <- function(res, prefix) {
  s <- res$sum_stats
  vals <- if (is.null(s)) rep(NA_real_, 6) else c(s$coef, s$se, s$z, s$p, s$coef_std, s$se_std)
  names(vals) <- paste0(prefix, c("_sum_coef", "_sum_se", "_sum_z", "_sum_p", "_sum_coef_std", "_sum_se_std"))
  out <- as.data.frame(as.list(vals))
  out[[paste0(prefix, "_sum_sign")]] <- if (is.null(s)) NA_character_ else ifelse(s$coef >= 0, "+", "-")
  out
}

arimax_for_row <- function(df, row) {
  resp_col <- if (row$direction == "culture_to_domain") "score" else paste0(row$predictor, "_z")
  caus_col <- if (row$direction == "culture_to_domain") paste0(row$predictor, "_z") else "score"

  y_raw <- df[[resp_col]]; x_raw <- df[[caus_col]]; gdp_raw <- df$gdp_z
  d <- row$diff_order
  if (d > 0) {
    y_d <- diff(y_raw, differences = d); x_d <- diff(x_raw, differences = d)
    gdp_d <- diff(gdp_raw, differences = d)
  } else { y_d <- y_raw; x_d <- x_raw; gdp_d <- gdp_raw }

  n_lags <- row$best_lag
  if (length(x_d) <= n_lags + 2) return(NULL)
  lbx <- build_lagged_xreg(x_d, n_lags)
  y_a <- y_d[lbx$valid_idx]; gdp_a <- gdp_d[lbx$valid_idx]

  basic <- perform_arimax(y_a, lbx$xreg)
  adj   <- perform_arimax(y_a, lbx$xreg, gdp_z = gdp_a)
  k <- row$granger_peak_lag
  xk <- lbx$xreg[, k, drop = FALSE]
  basic1 <- perform_arimax(y_a, xk)
  adj1   <- perform_arimax(y_a, xk, gdp_z = gdp_a)
  basic_sign <- sign_summary(basic$lag_coefs)
  adj_sign   <- sign_summary(adj$lag_coefs)

  summary_row <- data.frame(row[c("corpus","domain","predictor","direction","family")], n_lags = n_lags, diff_order = d,
    granger_p_fdr = row$p_fdr,
    basic_order = basic$order, basic_wald_p = basic$wald_p, basic_lb_p = basic$lb_p, basic_error = basic$error,
    basic_sign_pattern = basic_sign$pattern, 
    basic_n_pos = basic_sign$n_pos, basic_n_neg = basic_sign$n_neg,
    gdp_adj_order = adj$order, gdp_adj_wald_p = adj$wald_p, gdp_adj_lb_p = adj$lb_p, gdp_adj_error = adj$error,
    gdp_adj_sign_pattern = adj_sign$pattern, 
    gdp_adj_n_pos = adj_sign$n_pos, gdp_adj_n_neg = adj_sign$n_neg,
    stringsAsFactors = FALSE)
  summary_row <- cbind(summary_row, single_cols(basic1, "basic", k), single_cols(adj1, "gdp_adj", k),
                       sum_cols(basic, "basic"), sum_cols(adj, "gdp_adj"))

  lag_rows <- NULL
  if (!is.null(basic$lag_coefs)) {
    lag_rows <- cbind(row[rep(1, nrow(basic$lag_coefs)), c("corpus","domain","predictor","direction")],
                       model = "basic", basic$lag_coefs, row.names = NULL)
  }
  if (!is.null(adj$lag_coefs)) {
    adj_rows <- cbind(row[rep(1, nrow(adj$lag_coefs)), c("corpus","domain","predictor","direction")],
                       model = "gdp_adjusted", adj$lag_coefs, row.names = NULL)
    lag_rows <- if (is.null(lag_rows)) adj_rows else rbind(lag_rows, adj_rows)
  }

  list(summary = summary_row, lag_coefs = lag_rows)
}

# ---------------------------------------------------------------------------
cell_dfs <- list()
all_granger <- list(); i <- 1
for (corpus in CORPORA) for (dom in DOMS) {
  cat(sprintf("[%s] %s / %s\n", format(Sys.time(), "%H:%M:%S"), corpus, dom))
  df <- load_cell(corpus, dom)
  cell_dfs[[paste(corpus, dom)]] <- df
  all_granger[[i]] <- granger_for_cell(df, corpus, dom)
  i <- i + 1
}
granger_all <- do.call(rbind, all_granger)

granger_all$p_fdr <- NA_real_
for (co in CORPORA) for (dom in DOMS) for (dir in DIRECTIONS) for (fam in FDR_FAMILIES) {
  idx <- which(granger_all$corpus == co & granger_all$domain == dom &
               granger_all$direction == dir & granger_all$predictor %in% fam)
  granger_all$p_fdr[idx] <- p.adjust(granger_all$p_value[idx], method = "BH")
}
granger_all$significant <- granger_all$p_fdr < .05

arimax_rows <- list(); lag_coef_rows <- list(); k <- 1
sig <- granger_all[granger_all$significant, ]
for (i in seq_len(nrow(sig))) {
  r <- arimax_for_row(cell_dfs[[paste(sig$corpus[i], sig$domain[i])]], sig[i, ])
  if (!is.null(r)) {
    arimax_rows[[k]] <- r$summary
    if (!is.null(r$lag_coefs)) lag_coef_rows[[k]] <- r$lag_coefs
    k <- k + 1
  }
}
arimax_all <- if (length(arimax_rows)) do.call(rbind, arimax_rows) else NULL
lag_coefs_all <- if (length(lag_coef_rows)) do.call(rbind, lag_coef_rows) else NULL

empty_arimax <- function() data.frame(corpus=character(), domain=character(), predictor=character(),
  direction=character(), family=character(), n_lags=integer(), diff_order=integer(),
  granger_p_fdr=numeric(), basic_order=character(), basic_wald_p=numeric(), basic_lb_p=numeric(),
  basic_error=character(), basic_sign_pattern=character(),
  basic_n_pos=integer(), basic_n_neg=integer(),
  gdp_adj_order=character(), gdp_adj_wald_p=numeric(), gdp_adj_lb_p=numeric(), gdp_adj_error=character(),
  gdp_adj_sign_pattern=character(),
  gdp_adj_n_pos=integer(), gdp_adj_n_neg=integer(),
  basic_peak_lag=numeric(), basic_peak_coef=numeric(), basic_peak_coef_std=numeric(), basic_peak_se_std=numeric(),
  basic_peak_z=numeric(), basic_peak_p=numeric(), basic_peak_sign=character(),
  gdp_adj_peak_lag=numeric(), gdp_adj_peak_coef=numeric(), gdp_adj_peak_coef_std=numeric(), gdp_adj_peak_se_std=numeric(),
  gdp_adj_peak_z=numeric(), gdp_adj_peak_p=numeric(), gdp_adj_peak_sign=character(),
  basic_sum_coef=numeric(), basic_sum_se=numeric(), basic_sum_z=numeric(), basic_sum_p=numeric(),
  basic_sum_coef_std=numeric(), basic_sum_se_std=numeric(), basic_sum_sign=character(),
  gdp_adj_sum_coef=numeric(), gdp_adj_sum_se=numeric(), gdp_adj_sum_z=numeric(), gdp_adj_sum_p=numeric(),
  gdp_adj_sum_coef_std=numeric(), gdp_adj_sum_se_std=numeric(), gdp_adj_sum_sign=character(), stringsAsFactors=FALSE)
empty_lag_coefs <- function() data.frame(corpus=character(), domain=character(), predictor=character(),
  direction=character(), model=character(), lag=integer(), coefficient=numeric(), se=numeric(),
  z=numeric(), p=numeric(), coefficient_std=numeric(), se_std=numeric(), stringsAsFactors=FALSE)

write.csv(granger_all, file.path(OUT, "granger_fullcorpus.csv"), row.names = FALSE)
write.csv(if (!is.null(arimax_all)) arimax_all else empty_arimax(),
          file.path(OUT, "arimax_fullcorpus.csv"), row.names = FALSE)
write.csv(if (!is.null(lag_coefs_all)) lag_coefs_all else empty_lag_coefs(),
          file.path(OUT, "arimax_lag_coefficients_fullcorpus.csv"), row.names = FALSE)

cat("\n=== Granger FDR hits:", sum(granger_all$significant), "/", nrow(granger_all), "===\n")
print(granger_all[granger_all$significant, c("corpus","domain","predictor","direction","best_lag","p_value","p_fdr")])

if (!is.null(arimax_all)) {
  cat("\n=== ARIMAX follow-up (joint Wald p decides confirmation; peak-lag columns give direction/size) ===\n")
  print(arimax_all[, c("corpus","domain","predictor","direction","n_lags","basic_wald_p",
                        "basic_peak_lag","basic_peak_sign","basic_peak_coef_std","basic_sum_sign","basic_sum_coef_std","basic_sum_p","gdp_adj_wald_p")])
  cat(sprintf("\nsurvive basic (p<.05): %d/%d\n", sum(arimax_all$basic_wald_p < .05, na.rm=TRUE), nrow(arimax_all)))
  cat(sprintf("survive gdp-adjusted (p<.05): %d/%d\n", sum(arimax_all$gdp_adj_wald_p < .05, na.rm=TRUE), nrow(arimax_all)))
} else {
  cat("\nNo cells survived the Granger FDR gate -- no ARIMAX follow-up to report.\n")
}

cat("\nDone. Outputs in", OUT, "\n")
