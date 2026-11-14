| candidate | features | cv_roc_auc | val_roc_auc | val_pr_auc | val_recall | val_specificity | val_f2 | val_balanced_accuracy | search_seconds | best_params |
|---|---|---|---|---|---|---|---|---|---|---|
| hog_svc | hog | 0.8614 | 0.8607 | 0.8369 | 0.8522 | 0.6736 | 0.8028 | 0.7629 | 622.8900 | {"svc__C": 10, "svc__gamma": "scale"} |
| hog_logreg | hog | 0.8544 | 0.8566 | 0.8360 | 0.8522 | 0.6445 | 0.7967 | 0.7483 | 19.4700 | {"logreg__C": 0.0003} |
| hog_rf | hog | 0.8217 | 0.8379 | 0.8024 | 0.8522 | 0.6341 | 0.7946 | 0.7431 | 409.3200 | {"rf__max_depth": 20, "rf__min_samples_leaf": 5, "rf__n_estimators": 300} |
| hog_pca_hgb | hog | 0.8018 | 0.8093 | 0.7790 | 0.8522 | 0.5343 | 0.7745 | 0.6932 | 759.4100 | {"hgb__learning_rate": 0.1, "hgb__max_leaf_nodes": 31, "pca__n_components": 100} |
| raw_pca_svc | raw | 0.7800 | 0.7924 | 0.7382 | 0.8522 | 0.5509 | 0.7778 | 0.7016 | 70.0600 | {"pca__n_components": 200, "svc__C": 10, "svc__gamma": "scale"} |
| raw_pca_logreg | raw | 0.7856 | 0.7788 | 0.7460 | 0.8522 | 0.4823 | 0.7644 | 0.6673 | 132.0600 | {"logreg__C": 0.01, "pca__n_components": 200} |
