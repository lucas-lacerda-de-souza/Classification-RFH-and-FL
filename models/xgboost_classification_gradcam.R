# =========================================================
#  XGBoost Pipeline: Train / Validation / External Test
#  Multivariate Model Using All Nuclear Features
#  Unified Metric Plot (Train → Validation → External Test)
#  + Editable Section for Custom Metric Values
# =========================================================

# --- Load Required Libraries ---
packages <- c(
  "xgboost", "caret", "pROC", "MLmetrics", "dplyr",
  "ggplot2", "readxl", "scales", "reshape2"
)

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cran.r-project.org/")
    library(pkg, character.only = TRUE)
  }
}

# =========================================================
#  Load Datasets
# =========================================================
train_data <- read_excel("file.xlsx")
test_data  <- read_excel("file.xlsx")

features <- c(
  "Nucleus: Area",
  "Nucleus: Perimeter",
  "Nucleus: Circularity",
  "Nucleus: Eccentricity",
  "Nucleus: Hematoxylin OD mean"
)

train_data <- train_data[, c("Classe", features)]
test_data  <- test_data[,  c("Classe", features)]

# =========================================================
#  Split Dataset (80% Train / 20% Validation)
# =========================================================
set.seed(123)
train_index <- createDataPartition(train_data$Classe, p = 0.8, list = FALSE)

train_part <- train_data[train_index, ]
val_part   <- train_data[-train_index, ]

# --- Prepare Matrices ---
y_train <- train_part$Classe
X_train <- as.matrix(train_part[, -1])

y_val <- val_part$Classe
X_val <- as.matrix(val_part[, -1])

y_test <- test_data$Classe
X_test <- as.matrix(test_data[, -1])

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dval   <- xgb.DMatrix(data = X_val, label = y_val)
dtest  <- xgb.DMatrix(data = X_test, label = y_test)

watchlist <- list(train = dtrain, val = dval)

# =========================================================
#  Train XGBoost Model
# =========================================================
params <- list(
  objective = "binary:logistic",
  booster = "gbtree",
  eta = 0.01,
  max_depth = 8,
  subsample = 0.9,
  colsample_bytree = 0.9,
  eval_metric = "aucpr"
)

set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 3000,
  watchlist = watchlist,
  early_stopping_rounds = 100,
  verbose = 1
)

# =========================================================
#  Evaluation: Train / Validation / External Test
# =========================================================
get_metrics <- function(y_true, y_prob) {
  pred_labels <- ifelse(y_prob > 0.5, 1, 0)
  conf_mat <- confusionMatrix(as.factor(pred_labels), as.factor(y_true), positive = "1")
  
  data.frame(
    Accuracy  = conf_mat$overall["Accuracy"],
    Precision = Precision(y_pred = pred_labels, y_true = y_true),
    Recall    = Recall(y_pred = pred_labels, y_true = y_true),
    F1_Score  = F1_Score(y_pred = pred_labels, y_true = y_true),
    AUC       = auc(roc(y_true, y_prob))
  )
}

train_pred <- predict(xgb_model, X_train)
val_pred   <- predict(xgb_model, X_val)
test_pred  <- predict(xgb_model, X_test)

train_metrics <- get_metrics(y_train, train_pred)
val_metrics   <- get_metrics(y_val, val_pred)
test_metrics  <- get_metrics(y_test, test_pred)

all_metrics <- rbind(
  cbind(Set = "Training", train_metrics),
  cbind(Set = "Validation", val_metrics),
  cbind(Set = "External Test", test_metrics)
)

# =========================================================
#  Visualization
#  Ordered: Training → Validation → External Test
# =========================================================
dir.create("Results_XGBoost_AllFeatures", showWarnings = FALSE)

all_long <- reshape2::melt(all_metrics, id.vars = "Set",
                           variable.name = "Metric", value.name = "Value")

all_long$Set <- factor(all_long$Set,
                       levels = c("Training", "Validation", "External Test"))

p <- ggplot(all_long, aes(x = Metric, y = Value, fill = Set)) +
  geom_bar(stat = "identity",
           position = position_dodge(width = 0.8),
           width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", Value)),
            position = position_dodge(0.8),
            vjust = -0.5,
            size = 3.5) +
  scale_fill_manual(values = c("#2166AC", "#67A9CF", "#D1E5F0")) +
  ylim(0, 1.05) +
  labs(
    title = "XGBoost Performance (All Nuclear Features)",
    subtitle = "Training | Validation | External Test",
    x = "Metric",
    y = "Value",
    fill = "Dataset"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "top",
    legend.title = element_text(face = "bold")
  )

print(p)

ggsave(
  "Results_XGBoost_AllFeatures/all_datasets_metrics.png",
  p, width = 9, height = 5.5, dpi = 300
)
