# Metrics

Technicality components
- readability_grade: Flesch-Kincaid grade level
- avg_sentence_len: average words per sentence
- long_word_ratio: fraction of words with length >= 8
- jargon_rate: jargon terms per 100 words
- code_ratio: fraction of code-like characters

Technicality score
- For each domain, z-score each component across all assistant replies
- technicality = sum of z-scored components

Oscillation index per conversation
- mean_abs_delta = mean(|t_i - t_{i-1}|)
- flip_rate = fraction of sign changes in (t_i - t_{i-1})
- oscillation_index = mean_abs_delta * (1 + flip_rate)

Report
- Both components and the combined index
- Condition averages with standard error
