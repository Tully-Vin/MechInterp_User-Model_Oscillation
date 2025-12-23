# Mechanistic-lite probe

Goal
- Show that the model's internal representation of user expertise flips under oscillatory feedback

Method
- Build a labeled prompt set: expert vs novice user requests
- Run the model forward on each prompt with output_hidden_states
- Use the last token hidden state from a chosen layer as features
- Train a logistic regression probe

Application
- For each conversation, apply the probe at each user turn
- Plot expert probability over turns by condition

Notes
- This is not full mechanistic tracing, but gives internal evidence
- If time is tight, skip steering and report the probe only
