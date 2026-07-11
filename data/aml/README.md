# Data

This project uses the IBM "Transactions for Anti-Money Laundering" dataset (synthetic,
account-to-account transfers with per-transaction laundering labels and typologies).

Download from Kaggle (`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`) into
this folder, for example `HI-Small_Trans.csv` and `HI-Small_Patterns.txt`. With the
Kaggle CLI:

    kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv

Raw CSVs are git-ignored. With no data present, everything runs on a schema-faithful
mock. Point the loader at the real file and nothing downstream changes.
