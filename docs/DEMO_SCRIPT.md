# Two-Minute Demonstration Script

## 0:00–0:20 — Problem

“Fraud teams cannot investigate every mobile-money transaction. This project
asks how a model should prioritise transactions when the review team has a
limited capacity. It flags transactions for human review; it does not
automatically block payments.”

## 0:20–0:40 — Honest data scope

“I use PaySim, a public synthetic mobile-money dataset. No employer or customer
records are used. Synthetic performance does not prove that the model would
work on Tanzanian mobile-money transactions.”

Show the README's feature policy and say:

“I excluded the four balance columns identified by the publisher as leakage
risks. I also excluded account IDs and kept `isFlaggedFraud` out of the model,
evaluating it separately as the existing rule.”

## 0:40–1:00 — Evaluation design

“I split by complete hourly time steps. Models learn from the earliest period,
the threshold is selected on a later validation period, and the latest period
is kept untouched for the final test. Logistic regression and XGBoost receive
the same features and historical examples.”

## 1:00–1:30 — Review queue

“This control represents investigator capacity per 10,000 transactions. As I
increase it, the team catches more synthetic fraud, but also reviews more
genuine transactions. These slider changes are exploratory; they do not replace
the final validation-selected evaluation.”

## 1:30–1:45 — Individual prediction

“The score prioritises this transaction for review. It is a prediction—not
proof that the customer committed fraud.”

## 1:45–2:00 — Comparison

“I compare precision, recall, average precision, false alerts, missed fraud,
runtime and memory. The contribution is connecting model evaluation to the
review capacity and customer impact a fraud team must manage.”

