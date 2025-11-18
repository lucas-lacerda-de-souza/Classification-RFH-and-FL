# ================================================================
#  XGBoost Classification Pipeline
#  Train/Test Split • Model Evaluation • ROC Curve • SHAP Analysis
# ================================================================

# ---------------------------
# Install required packages
# ---------------------------
install.packages(c(
  "xgboost", "SHAPforxgboost", "dplyr", "ggplot2",
  "readxl", "caret", "pROC", "reshape2"
))

# ---------------------------
# Load libraries
# ---------------------------
library(xgboost)
library(SHAPforxgboost)
library(dplyr)
library(ggplot2)
library(readxl)
library(caret)
library(pROC)
library(reshape2)

# ================================================================
# 1. Load dataset
# ================================================================
file_path <- "file.xlsx"
data <- read_excel(file_path)

# ================================================================
# 2. Preprocessing
# ================================================================
colnames(data)[1] <- "Classes"
predictors <- setdiff(colnames(data), "Classes")

data_clean <- data %>%
  mutate(across(all_of(predictors), as.numeric))

set.seed(123)
train_index <- sample(seq_len(nrow(data_clean)), size = 0.7 * nrow(data_clean))
train_data <- data_clean[train_index, ]
test_data  <- data_clean[-train_index, ]

X_train <- as.matrix(train_data %>% select(all_of(predictors)))
y_train <- as.factor(train_data$Classes)

X_test  <- as.matrix(test_data %>% select(all_of(predictors)))
y_test  <- as.factor(test_data$Classes)

# Ensure classes are labeled "1" and "2"
levels(y_train) <- c("1", "2")
levels(y_test)  <- c("1", "2")

# ================================================================
# 3. Train XGBoost model
# ================================================================
model <- xgboost(
  data = X_train,
  label = as.numeric(y_train) - 1,
  objective = "binary:logistic",
  nrounds = 100,
  eta = 0.1,
  max_depth = 6,
  verbose = 0
)

# ================================================================
# 4. Model Evaluation
# ================================================================
y_pred <- predict(model, X_test)
y_pred_class <- as.factor(ifelse(y_pred > 0.5, "2", "1"))

confusion <- confusionMatrix(y_pred_class, y_test, positive = "2")
accuracy  <- confusion$overall["Accuracy"]
precision <- confusion$byClass["Precision"]
recall    <- confusion$byClass["Recall"]
f1        <- confusion$byClass["F1"]

roc_curve <- roc(as.numeric(y_test) - 1, y_pred)
auc_value <- auc(roc_curve)

cat("Model Metrics:\n")
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-Score:", f1, "\n")
cat("AUC:", auc_value, "\n")

metrics_df <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1-Score", "AUC"),
  Value = c(
    as.numeric(accuracy),
    as.numeric(precision),
    as.numeric(recall),
    as.numeric(f1),
    as.numeric(auc_value)
  )
)

# ================================================================
# 5. Metric Bar Plot (No Colors)
# ================================================================
ggplot(metrics_df, aes(x = Metric, y = Value)) +
  geom_col(width = 0.7, color = "black", fill = "gray80") +
  geom_text(aes(label = round(Value, 3)), vjust = -0.3, size = 4) +
  labs(
    title = "XGBoost Model Performance",
    x = "Metric",
    y = "Value"
  ) +
  ylim(0, 1) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    axis.text = element_text(size = 12)
  )

# ================================================================
# 6. ROC Curve (No Colors)
# ================================================================
y_true <- as.numeric(y_test) - 1
y_scores <- y_pred

roc_pos <- roc(y_true, y_scores)
roc_neg <- roc(1 - y_true, 1 - y_scores)

plot(roc_pos, col = "black", lwd = 2,
     main = "ROC Curves for Both Classes", legacy.axes = TRUE)
lines(roc_neg, col = "gray40", lwd = 2, lty = 2)

abline(a = 0, b = 1, lty = 3, col = "gray60")

text(0.65, 0.25, paste("AUC (Positive Class) =", round(auc(roc_pos), 3)), cex = 1.1)
text(0.65, 0.18, paste("AUC (Negative Class) =", round(auc(roc_neg), 3)), cex = 1.1)

legend(
  "bottomright",
  legend = c("Positive Class", "Negative Class"),
  col = c("black", "gray40"),
  lwd = 2,
  lty = c(1, 2),
  cex = 1
)

# ================================================================
# 7. SHAP Values
# ================================================================
shap_values <- shap.values(xgb_model = model, X_train = X_train)
shap_long   <- shap.prep(xgboost_model = model, X_train = X_train)

# SHAP Summary Plot (No custom colors)
shap.plot.summary(shap_long) +
  scale_color_gradient(low = "gray60", high = "black") +
  theme(
    axis.text.y = element_text(size = 10),
    axis.text.x = element_text(size = 14),
    plot.title  = element_text(size = 16, face = "bold")
  )

# ================================================================
# 8. SHAP Decision Plot (No Colors)
# ================================================================
cumulative_shap <- as.data.frame(t(apply(shap_values$shap_score, 1, cumsum)))
colnames(cumulative_shap) <- predictors
cumulative_shap$Sample <- 1:nrow(cumulative_shap)

cumulative_shap_melted <- melt(cumulative_shap, id.vars = "Sample")

ggplot(cumulative_shap_melted, aes(x = Sample, y = value)) +
  geom_line(size = 0.7, alpha = 0.8, color = "black") +
  labs(
    title = "Decision Plot (Cumulative SHAP Values)",
    x = "Sample",
    y = "Cumulative SHAP Value"
  ) +
  theme_minimal()
