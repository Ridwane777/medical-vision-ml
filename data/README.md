# Data

This folder is git-ignored except for this file. Put the datasets here; every command and notebook uses these
default locations.

```
data/
├── knee/{train,val,test}/{0,1,2,3,4}/*.png
└── testimonies.csv
```

## Knee X-rays: `data/knee/`

Download the Kaggle [Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)
(license CC BY 4.0) and copy its split folders into `data/knee/`. Each split has one sub-folder per
Kellgren-Lawrence grade (`0` to `4`) holding 224x224 grayscale PNGs. An extra `auto_test/` split is ignored unless listed
in `extra_test_splits` of the config.

If the images live elsewhere, pass `--data /path/to/root` to `python -m kneeclf`, or set the environment variable
`KNEE_DATA_ROOT` for the notebook. Details: [data card](../docs/data-cards/knee-xrays.md).

## Patient testimonies: `data/testimonies.csv`

The CSV is not distributed: we have neither its license nor permission to redistribute it.

Expected format of the original file: UTF-8 (with BOM), comma separated, **no header**, three columns: an `ID`, a `Code`
column (always `0x000000`, sometimes missing, which shifts the row to two fields) and the free English text. About 1,011
rows. The pipeline does not depend on this layout: any CSV with a text column works (delimiter, header, encoding and
columns are detected). Details: [data card](../docs/data-cards/patient-testimonies.md).
