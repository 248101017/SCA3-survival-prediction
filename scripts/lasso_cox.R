#!/usr/bin/env Rscript
# ============================================================
# LASSO-Cox variable screening (glmnet)
# ============================================================
# Performs penalised Cox regression with a LASSO penalty on
# pre-processed, academically named candidate predictors.
#
# Inputs
# ------
#   - CSV exported by data_preprocessing.py
#     (imputed features with academic column names)
#
# Outputs
# -------
#   - Selected variables at lambda.min and lambda.1se (CSV + TXT)
#   - Full coefficient table at both thresholds (CSV)
#   - Combined figure: coefficient profiles + CV curve (PNG/PDF)
#   - Standalone coefficient profiles figure (PNG/PDF)
#   - Standalone CV curve figure (PNG/PDF)
#
# Usage
# -----
#   Rscript lasso_cox.R                           # uses defaults
#   Rscript lasso_cox.R --input data.csv --outdir results/lasso
#
# Dependencies
# ------------
#   readr, dplyr, glmnet, survival
# ============================================================

suppressMessages(library(readr))
suppressMessages(library(dplyr))
suppressMessages(library(glmnet))
suppressMessages(library(survival))

# ── Command-line arguments (with defaults) ─────────────────
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default) {
  idx <- which(args == flag)
  if (length(idx) > 0 && idx < length(args)) return(args[idx + 1])
  return(default)
}

in_csv  <- get_arg("--input",  "outputs/lasso/features_imputed_academic.csv")
out_dir <- get_arg("--outdir", "outputs/lasso")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("[INFO] input  :", in_csv, "\n")
cat("[INFO] outdir :", out_dir, "\n")

# ── 1. Load data ───────────────────────────────────────────
data0 <- read_csv(in_csv, show_col_types = FALSE)

time_col  <- "Survival_time"
event_col <- "status"

feature_cols <- c(
  "Sex",
  "BMI",
  "Disease duration",
  "Long CAG repeats",
  "SARA score",
  "PHQ-9 depression",
  "EQ-VAS",
  "INAS count",
  "GAD-7 anxiety",
  "INAS_Hyperreflexia",
  "INAS_Arreflexia",
  "INAS_Extensor plantar response",
  "INAS_Spasticity",
  "INAS_Paresis",
  "INAS_Muscle atrophy",
  "INAS_Fasciculations",
  "INAS_Myoclonus",
  "INAS_Dystonia",
  "INAS_Sensory symptoms",
  "INAS_Urinary dysfunction",
  "INAS_Cognitive impairment",
  "INAS_Oculomotor signs"
)

all_needed <- c(time_col, event_col, feature_cols)
missing <- setdiff(all_needed, colnames(data0))
if (length(missing) > 0) {
  cat("[ERROR] Required columns not found:\n")
  cat(paste0("  - ", missing, collapse = "\n"), "\n")
  stop("Aborting: required columns missing (check preprocessing output).")
}
data <- data0 %>% select(all_of(all_needed))

# ── 2. Type conversion ────────────────────────────────────
binary_vars <- intersect(
  c("Sex",
    "INAS_Hyperreflexia", "INAS_Arreflexia",
    "INAS_Extensor plantar response",
    "INAS_Spasticity", "INAS_Paresis", "INAS_Muscle atrophy",
    "INAS_Fasciculations", "INAS_Myoclonus", "INAS_Dystonia",
    "INAS_Sensory symptoms", "INAS_Urinary dysfunction",
    "INAS_Cognitive impairment", "INAS_Oculomotor signs"),
  colnames(data)
)

num_vars <- intersect(
  c("BMI", "Disease duration", "Long CAG repeats",
    "SARA score", "PHQ-9 depression", "EQ-VAS",
    "INAS count", "GAD-7 anxiety"),
  colnames(data)
)

data[[event_col]] <- as.numeric(data[[event_col]])
data[[event_col]] <- ifelse(data[[event_col]] > 0, 1, 0)

data <- data %>%
  mutate(
    across(all_of(binary_vars), ~ as.numeric(as.factor(.)) - 1),
    across(all_of(num_vars), as.numeric),
    across(all_of(time_col), as.numeric)
  )

# ── 3. Remove invalid rows ────────────────────────────────
data <- data %>%
  filter(.data[[time_col]] > 0, .data[[event_col]] %in% c(0, 1)) %>%
  na.omit()

if (nrow(data) < 10) stop("Too few samples after cleaning.")
cat("[INFO] samples used:", nrow(data), "\n")

# ── 4. Design matrix and Surv object ──────────────────────
X <- model.matrix(
  ~ .,
  data = data %>% select(-all_of(c(time_col, event_col)))
)[, -1, drop = FALSE]

y <- Surv(data[[time_col]], data[[event_col]])

# ── 5. Fit LASSO-Cox with 10-fold CV ─────────────────────
set.seed(42)
lasso_fit <- cv.glmnet(
  x = X, y = y,
  family = "cox",
  alpha = 1,
  nfolds = 10,
  standardize = TRUE
)

cat("lambda.min =", lasso_fit$lambda.min, "\n")
cat("lambda.1se =", lasso_fit$lambda.1se, "\n\n")

writeLines(
  c(paste0("lambda.min = ", lasso_fit$lambda.min),
    paste0("lambda.1se = ", lasso_fit$lambda.1se)),
  con = file.path(out_dir, "lasso_lambdas.txt")
)

# ── 6. Extract selected variables ─────────────────────────
get_selected <- function(model, s) {
  coef_mat <- coef(model, s = s)
  idx <- which(as.numeric(coef_mat) != 0)
  if (length(idx) == 0)
    return(data.frame(Variable = character(0),
                      Coefficient = numeric(0)))
  out <- data.frame(
    Variable    = gsub("`", "", rownames(coef_mat)[idx]),
    Coefficient = as.numeric(coef_mat)[idx],
    stringsAsFactors = FALSE
  )
  out <- out[out$Variable != "(Intercept)", , drop = FALSE]
  out %>% arrange(desc(abs(Coefficient)))
}

vars_1se <- get_selected(lasso_fit, "lambda.1se")
vars_min <- get_selected(lasso_fit, "lambda.min")

cat("=== Selected at lambda.1se ===\n"); print(vars_1se)
cat("\n=== Selected at lambda.min ===\n"); print(vars_min)

write_csv(vars_1se, file.path(out_dir, "selected_lambda1se.csv"))
write_csv(vars_min, file.path(out_dir, "selected_lambdamin.csv"))

# Full coefficient table
coef_1se <- as.matrix(coef(lasso_fit, s = "lambda.1se"))
coef_min <- as.matrix(coef(lasso_fit, s = "lambda.min"))
coef_all <- data.frame(
  feature         = gsub("`", "", rownames(coef_1se)),
  coef_lambda_1se = as.numeric(coef_1se),
  coef_lambda_min = as.numeric(coef_min),
  stringsAsFactors = FALSE
)
write_csv(coef_all, file.path(out_dir, "lasso_coefficients_all.csv"))

# Text summary
txt_path <- file.path(out_dir, "lasso_selected_summary.txt")
lines <- c(
  "LASSO-Cox selected variables",
  sprintf("lambda.min = %s", lasso_fit$lambda.min),
  sprintf("lambda.1se = %s", lasso_fit$lambda.1se),
  "",
  "--- lambda.1se ---",
  if (nrow(vars_1se) == 0) "  (none)"
  else paste0("  ", vars_1se$Variable,
              " (", sprintf("%.6f", vars_1se$Coefficient), ")"),
  "",
  "--- lambda.min ---",
  if (nrow(vars_min) == 0) "  (none)"
  else paste0("  ", vars_min$Variable,
              " (", sprintf("%.6f", vars_min$Coefficient), ")")
)
writeLines(lines, con = txt_path)

# ── 7. Plotting helpers ──────────────────────────────────
save_figure <- function(png_path, pdf_path,
                        width_px, height_px, res,
                        width_in, height_in, draw_fn) {
  png(png_path, width = width_px, height = height_px, res = res)
  draw_fn(); dev.off()
  cat("[INFO] Saved:", png_path, "\n")
  
  tryCatch({
    pdf(pdf_path, width = width_in, height = height_in)
    draw_fn(); dev.off()
    cat("[INFO] Saved:", pdf_path, "\n")
  }, error = function(e)
    cat("[WARN] PDF export skipped:", conditionMessage(e), "\n"))
}

plot_margins <- function() {
  par(mar = c(4.8, 4.8, 4.2, 1.0),
      mgp = c(2.6, 0.8, 0), tcl = -0.25,
      cex.axis = 1.05, cex.lab = 1.15, cex.main = 1.15)
}

draw_coef_path <- function() {
  plot_margins()
  plot(lasso_fit$glmnet.fit, xvar = "lambda",
       label = FALSE, lwd = 1.3, xaxt = "n", xlab = "", main = "")
  axis(1)
  mtext(expression(-log(lambda)), side = 1, line = 2.7, cex = 1.05)
  mtext("LASSO\u2013Cox coefficient profiles",
        side = 3, line = 2.3, font = 2)
  abline(v = log(lasso_fit$lambda.min),
         col = "#D62728", lty = 2, lwd = 1.6)
  abline(v = log(lasso_fit$lambda.1se),
         col = "#1F77B4", lty = 2, lwd = 1.6)
  legend("topright",
         legend = c("lambda.min", "lambda.1se"),
         col = c("#D62728", "#1F77B4"),
         lty = 2, lwd = 1.6, bty = "n", cex = 0.95)
}

draw_cv_curve <- function() {
  plot_margins()
  plot(lasso_fit, pch = 16, col = "#D62728",
       cex = 0.9, lwd = 1.2, main = "",
       xlab = expression(-log(lambda)))
  mtext("Ten-fold cross-validation for LASSO\u2013Cox model",
        side = 3, line = 2.3, font = 2)
}

draw_combined <- function() {
  op <- par(mfrow = c(1, 2)); on.exit(par(op))
  draw_coef_path(); draw_cv_curve()
}

# ── 8. Export figures ─────────────────────────────────────
save_figure(
  file.path(out_dir, "lasso_combined.png"),
  file.path(out_dir, "lasso_combined.pdf"),
  2600, 1100, 220, 12, 5, draw_combined)

save_figure(
  file.path(out_dir, "lasso_coef_path.png"),
  file.path(out_dir, "lasso_coef_path.pdf"),
  1600, 1100, 220, 6.5, 5.5, draw_coef_path)

save_figure(
  file.path(out_dir, "lasso_cv_curve.png"),
  file.path(out_dir, "lasso_cv_curve.pdf"),
  1600, 1100, 220, 6.5, 5.5, draw_cv_curve)

cat("\n[DONE] All outputs saved to:", out_dir, "\n")