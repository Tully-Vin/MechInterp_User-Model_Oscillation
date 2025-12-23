# Experiment design

Domains
- Photoshop: color and lighting match between subject and background
- Python: traceback debug and fix

Conditions
- consistent_basic: user keeps asking for simpler answers
- consistent_technical: user keeps asking for more technical answers
- oscillatory: user alternates between the two
- anchored: oscillatory feedback but with a stable preference statement

Conversation structure
- System prompt
- Base user prompt (domain specific)
- Assistant response
- User feedback 1
- Assistant response
- User feedback 2
- Assistant response
- User feedback 3
- Assistant response

Default run size
- 2 domains x 4 conditions x 10 conversations x 4 assistant turns

Primary outcomes
- Oscillation index (technicality swings)
- Technicality over turns by condition
- Stability gain for anchored condition
