#!/usr/bin/env python3
"""Build the Lending Club sample with a natural client structure (borrower's US state) into ../data/external/lending_club.csv.

Source: Lending Club accepted loans 2007-2018Q4 (CC0), mirrored on Hugging Face as codesignal/lending-club-loan-accepted
(1.7 GB). The script streams the file (or reads a local copy given as the first argument), keeps loans issued in 2013 and
2014 that ended Fully Paid or Charged Off/Default, keeps applicant-side columns only (no lender-assigned grade, interest
rate or payment outcomes), and draws 1,500 loans from each of California, New York, Texas and Florida (seed 42).
Usage: python fetch_lending_club.py [path/to/accepted_2007_to_2018Q4.csv]
"""
import sys, pathlib, numpy as np, pandas as pd
URL = "https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/resolve/main/accepted_2007_to_2018Q4.csv"
STATES, PER = ["CA", "NY", "TX", "FL"], 1500
COLS = ["loan_amnt", "term", "emp_length", "home_ownership", "annual_inc", "verification_status", "issue_d", "loan_status",
        "purpose", "addr_state", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc",
        "pub_rec", "revol_bal", "revol_util", "total_acc"]
src = sys.argv[1] if len(sys.argv) > 1 else URL
parts = []
for ch in pd.read_csv(src, usecols=COLS, chunksize=200_000, low_memory=False):
    ch = ch[ch.loan_status.isin(["Fully Paid", "Charged Off", "Default"]) & ch.addr_state.isin(STATES)]
    ch = ch[ch.issue_d.astype(str).str[-4:].isin(["2013", "2014"])]
    parts.append(ch)
df = pd.concat(parts, ignore_index=True)
df["bad"] = (df.loan_status != "Fully Paid").astype(int)
df = pd.concat([df[df.addr_state == st].sample(PER, random_state=42) for st in STATES], ignore_index=True)
out = pathlib.Path(__file__).resolve().parent.parent / "data" / "external"; out.mkdir(parents=True, exist_ok=True)
df.drop(columns=["loan_status", "issue_d"]).to_csv(out / "lending_club.csv", index=False)
print(df.groupby("addr_state").bad.agg(["size", "mean"]))
