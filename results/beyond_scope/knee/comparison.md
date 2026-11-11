| candidate | features | cv_roc_auc | val_roc_auc | val_pr_auc | val_recall | val_specificity | val_f2 | val_balanced_accuracy | search_seconds | best_params |
|---|---|---|---|---|---|---|---|---|---|---|
| resnet50_logreg | resnet50 | 0.8374 | 0.8394 | 0.8090 | 0.8522 | 0.6258 | 0.7929 | 0.7390 | 25.2000 | {"logreg__C": 0.001} |
| efficientnet_b0_logreg | efficientnet_b0 | 0.8532 | 0.8384 | 0.8059 | 0.8522 | 0.6071 | 0.7890 | 0.7296 | 9.3000 | {"logreg__C": 0.001} |
| efficientnet_b0_svc | efficientnet_b0 | 0.8538 | 0.8371 | 0.8027 | 0.8522 | 0.6320 | 0.7942 | 0.7421 | 161.2000 | {"svc__C": 10, "svc__gamma": 0.0001} |
| resnet50_svc | resnet50 | 0.8339 | 0.8318 | 0.8035 | 0.8522 | 0.6133 | 0.7903 | 0.7327 | 258.6000 | {"svc__C": 1, "svc__gamma": 0.0001} |
