# Model choice

Tested qwen3:4b and gemma3:4b locally on 6 questions, with no sources.

Findings
- Both made serious errors, including denying hydroxycarbamide exists and giving dose information.
- Both followed "answer only from this passage" instructions.
- Local answers took 2 to 8 minutes on an 8 GB laptop.

Decision
- Answers must come only from retrieved sources (RAG).
- Emergencies and doses are handled by fixed rules, not the model.
- Development: gemma3:4b locally. Live app: a larger hosted model (being tested next).