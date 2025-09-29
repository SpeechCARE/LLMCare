# Component 5 – External Generalizability Evaluation (DementiaBank Delaware Corpus)

## Overview

This component evaluates the **generalizability** of our speech-based screening pipeline on the **DementiaBank Delaware corpus**, an independent dataset focused on **mild cognitive impairment (MCI) vs controls**.

* **Dataset:** 205 English-speaking participants

  * 99 with clinically diagnosed MCI
  * 106 controls
* **Tasks:**

  * Picture descriptions (Cookie Theft, Cat Rescue, Rockwell)
  * Cinderella story recall
  * Procedural discourse
* **Labels:** Binary (MCI vs control)
* **Splits:**

  * ~60% training (n=124)
  * ~20% validation (n=40)
  * ~20% test (n=41)
  * **Note:** Each participant appears in only one partition.

Initial experiments showed Cat Rescue and Rockwell provided limited discriminatory signals. Final evaluations therefore focused on **Cookie Theft**, **Cinderella recall**, and **procedural discourse**, which yielded stronger performance.

---

##  Relation to Other Components

This component **reuses the same pipeline as Component 1 (ADReSSo experiments)**:

* **Transformer baselines**
* **Handcrafted linguistic features** (110 lexical, syntactic, semantic, psycholinguistic).
* **Fusion classifier** combining transformer embeddings + features.

👉 The **only change required** is updating the **data paths** in the configuration files to point to the **Delaware dataset** instead of ADReSSo.
