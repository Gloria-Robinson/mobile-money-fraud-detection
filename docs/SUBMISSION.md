# Submission Description

## Title

**Beyond Accuracy: Review-Capacity-Aware Fraud Detection for Mobile-Money Transactions**

## One-sentence pitch

I am building a human-in-the-loop mobile-money fraud prototype that asks not
only whether fraud can be predicted, but how many useful alerts a constrained
investigation team can realistically review.

## Abstract

Mobile-money fraud models are often evaluated using overall predictive scores
without considering whether a fraud operations team can investigate the alerts
they generate. This project develops a review-capacity-aware fraud triage
prototype using the public PaySim synthetic mobile-money dataset.

The system assigns risk scores and prioritises transactions for human review
rather than automatically blocking payments. Logistic regression and XGBoost
are compared with PaySim's existing `isFlaggedFraud` rule. Transactions are
divided chronologically so models learn from earlier activity and are evaluated
on later, untouched transactions.

To reduce leakage, the four balance columns identified by the publisher are
excluded. Account identifiers and `isFlaggedFraud` are also excluded from model
inputs. Thresholds are selected using validation data under a declared review
capacity, measured as transactions reviewed per 10,000.

Evaluation focuses on precision, fraud recall, average precision, false alerts,
missed fraud, runtime, memory and review workload—not accuracy alone. An
interactive dashboard shows how review capacity affects both customers and
investigators.

PaySim is synthetic. This is an educational research prototype, not evidence of
performance on Tanzanian mobile-money data, and it must not block real payments.

## Current completion statement

> The reproducible experiment and dashboard have been implemented and tested
> locally. In a disclosed 500,000-row training baseline, XGBoost achieved 35.53%
> precision and 40.02% recall on the untouched synthetic test period. The
> validation-selected threshold also produced more test alerts than its intended
> review budget, revealing review-workload drift that should be monitored over
> time. These results are from PaySim and do not establish real-world performance.

