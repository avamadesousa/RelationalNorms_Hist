#!/usr/bin/env Rscript
# Bayesian change point detection 
# bcp() returns a posterior probability of a change point AT each
# timepoint. Reports the top-N years by that posterior probability,
# matching the paper's own reporting style ("the first three highest
# probabilities determined by Bayesian Change Point Detection").
#
# Usage:
#   Rscript bcp_real.R --input <csv> --output-dir <dir> [--top-n <N>]
#
# Input CSV columns required: year, score (or 'mean', auto-detected)

suppressPackageStartupMessages({
  library("optparse")
})

option_list <- list(
  make_option(c("--input"), type="character", default=NULL, metavar="FILE"),
  make_option(c("--output-dir"), type="character", default=NULL, metavar="DIR"),
  make_option(c("--top-n"), type="integer", default=5, metavar="N")
)
opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)
if (is.null(opt$input)) stop("Error: --input is required")
if (is.null(opt$`output-dir`)) stop("Error: --output-dir is required")

suppressPackageStartupMessages({
  library("bcp")
})

output_dir <- opt$`output-dir`
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

df <- read.csv(opt$input)
score_col <- if ("score" %in% colnames(df)) "score" else if ("mean" %in% colnames(df)) "mean" else stop("need a 'score' or 'mean' column")
df <- df[order(df$year), ]

label <- tools::file_path_sans_ext(basename(opt$input))
safe_label <- gsub("[^A-Za-z0-9_]", "_", label)

set.seed(42)
fit <- bcp(df[[score_col]])
post_prob <- fit$posterior.prob
# bcp's posterior.prob has length n; the last element is often NA
# (no "change" defined past the final point) -- drop before ranking
valid <- !is.na(post_prob)

result <- data.frame(
  year = df$year[valid],
  score = df[[score_col]][valid],
  posterior_prob = post_prob[valid]
)
result <- result[order(-result$posterior_prob), ]

top_n <- min(opt$`top-n`, nrow(result))
top_result <- head(result, top_n)
top_result$rank <- seq_len(top_n)

out_path <- file.path(output_dir, paste0(safe_label, "_bcp_top", top_n, ".csv"))
write.csv(top_result, out_path, row.names = FALSE)

full_path <- file.path(output_dir, paste0(safe_label, "_bcp_posterior_full.csv"))
write.csv(result[order(result$year), ], full_path, row.names = FALSE)

cat("Saved:", out_path, "\n")
print(top_result[, c("rank", "year", "posterior_prob")])
 