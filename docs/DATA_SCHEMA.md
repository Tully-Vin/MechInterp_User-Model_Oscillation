# Data schema

Raw outputs (JSONL)
- run_id: string
- domain: string
- condition: string
- conversation_id: string
- seed: int
- messages: list of {role, content}
- gen_config: dict

Analysis outputs (CSV)
- domain, condition, conversation_id, turn_index
- technicality components
- technicality score

Probe outputs (CSV)
- domain, condition, conversation_id, user_turn_index
- expert_probability
