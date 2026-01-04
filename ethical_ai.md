# Ethical AI in Healthcare — Checklist and Notes

This document provides a short checklist and resources to help build responsibly in healthcare projects.

Key principles
- **Safety:** Avoid providing clinical diagnoses, ensure UI clearly states limitations and encourages users to seek professional care.
- **Privacy:** Minimize PHI, pre-process / de-identify data, avoid logging raw patient-identifiable information.
- **Fairness:** Evaluate model performance across demographic groups (age, gender, ethnicity) and mitigate disparate impact.
- **Transparency:** Provide model cards, data provenance, and explanation tools (SHAP, LIME) for clinicians.
- **Robustness:** Test models on edge cases and noisy inputs; use conservative defaults when confidence is low.

Practical checklist
- [ ] Data governance: documented data sources, retention policy, and access controls.
- [ ] Consent & purpose: ensure data was collected with appropriate consent and the ML purpose is documented.
- [ ] De-identification: remove names, MRNs, and other direct identifiers before persistence.
- [ ] Performance by subgroup: compute metrics (sensitivity, specificity, AUC) for subpopulations.
- [ ] Explainability: integrate SHAP/LIME outputs into clinician-facing views for high-impact predictions.
- [ ] Human-in-the-loop: require clinician review for high-risk recommendations.
- [ ] Logging & monitoring: log inputs/outputs (privacy-preserving), monitor model drift and data distribution.
- [ ] Incident response: maintain a plan to rollback models and notify stakeholders on errors/harm.

Further reading & resources
- Model Cards: Mitchell et al., "Model Cards for Model Reporting".
- Datasheets for Datasets: Gebru et al., "Datasheets for Datasets".
- WHO guidance on AI in health: https://www.who.int/publications/
- Papers and best practices from major health ML groups (e.g., Stanford, NHSX).

How to use this repository
- Add an `ethical_ai.md` summary to model release artifacts when saving models in `saved_models/`.
- Use `healthai/explainability.py` to generate explanation artifacts and include them with model reports.
