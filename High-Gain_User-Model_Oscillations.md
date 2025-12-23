# **High-Gain User-Model Oscillations in Conditional LLM Generation**

---

# **1\. Hypothesis**

This hypothesis proposes that large language models can exhibit **high-gain instability** in conditional generation when **inferring user expertise** for users whose communication style or knowledge spans **multiple latent categories**.

This instability shows up as rapid oscillation between **incompatible response modes**, for example alternating between *assuming shared technical context* and *assuming minimal subject knowledge*. It is most noticeable when prompts are **oppositionally structured**, such that successive turns are dominated by corrective or negative feedback without clear positive anchoring.

In these cases, the model appears to **repeatedly revise its assumptions** in an attempt to satisfy the user’s preferences. When those preferences are underspecified or expressed **primarily through rejection**, the model **overcorrects**, resulting in **oscillating behaviour** *rather than gradual convergence*.

The effect may arise from a combination of representational, decoding, training-distribution, and interface-level mechanisms. Regardless of where it originates, the claim is that this instability is **plausibly mechanistically identifiable**, **causally relevant** to observable output behaviour, and disproportionately affects users who do not align cleanly with **dominant training archetypes**.

The phenomenon is unlikely to be well explained by any single isolated mechanism. **Mechanistic interpretability** offers a plausible route to isolating contributing dynamics and identifying ways to **improve stability and accessibility** *without changing* the model’s underlying knowledge or capabilities.

This work focuses on **expertise inference** as one concrete case of a broader class of user-model effects. A substantial body of prior research has shown that inferred or implied user characteristics can systematically influence model outputs, including **salary advice**, **medical recommendations**, and **evaluative judgments** *(e.g. audit studies of demographic bias in AI systems and language models; Obermeyer et al., 2019; Bender et al., 2021\)*.

Findings from this exploration are *not* assumed to generalise automatically. The motivation is that **shared mathematical influences on priors**, **decoding preferences**, and **output grammars** may underlie *multiple* forms of problematic user-model behaviour.

---

# **2\. The Behavioural Failure Mode I Keep Seeing**

When a user alternates between precise technical language and plain-language corrections, the model tends to swing its response style sharply rather than settling on a **consistent level**.

Instead of converging on a **stable abstraction level**, the model repeatedly resets its assumptions about what the user knows, often overcorrecting in response to minor feedback.

When feedback is **primarily negative or corrective**, the model experiments with **extreme shifts** in tone or register *rather than making small, local adjustments*.

This creates friction in which the user feels talked down to, talked past, or required to continually restate how they want to be addressed.

Over time, the interaction shifts away from the actual topic and toward managing the model’s attempts to satisfy the user, **increasing cognitive load** and **reducing effective usefulness**.

---

# **3\. A Working Sketch of the Underlying Dynamics**

## **3.1 Claim 1: Implicit User Modelling**

* Inferred subject-matter expertise  
* Preferred abstraction level  
* Tolerance for formality vs informality  
* Responsiveness to correction and feedback  
* Perceived expectations about helpfulness or tone

## **3.2 Claim 2: Conditioning of Generation**

* Biasing which response styles are treated as appropriate  
* Influencing how much explanation or justification is produced  
* Shaping assumptions about shared context  
* Weighting stylistic and social constraints during decoding

## **3.3 Claim 3: Chain-of-Thought as a Gain Amplifier**

* Amplifying early assumptions about the user across the full response  
* Increasing commitment to an inferred user model within a single turn  
* Favouring replacement of prior assumptions over gradual adjustment

## **3.4 Claim 4: Instability Under Conflicting Feedback**

* The inferred user model shifts sharply between incompatible states  
* Response style oscillates *instead of converging*  
* Small corrections trigger large changes in tone or abstraction  
* The model fails to stabilise due to repeated re-inference rather than refinement

---

# **4\. Why This Isn’t Already Addressed**

Most existing explanations frame these failures in terms of **content quality** or **safety trade-offs**, attributing degradation to **missing knowledge**, **hallucination**, or **instruction-following errors**.

These explanations assume the model’s estimate of the user is **stable or slowly varying**, and that feedback **monotonically improves** **alignment**.

This assumption fails when feedback is sparse or oppositional, causing repeated re-inference *rather than refinement* and leading to oscillation.

---

# **5\. A North Star for What This Explains**

**If I’m right:**

* Oscillation can be reduced by stabilising user expertise inference  
* Large swings after minor feedback disappear  
* Systems can distinguish content rejection from framing rejection  
* Stability improves without changing model competence  
* Instability correlates with feedback structure rather than topic difficulty

**If I’m wrong:**

* Behaviour is explained by per-turn prompt ambiguity  
* Damping fails due to content uncertainty or safety constraints  
* Problems persist despite clearly scoped feedback  
* Improvements require capability or data changes  
* Instability correlates with domain complexity rather than interaction pattern

---

# **6\. Scope Control and Non-Claims**

* This is *not* a claim that models lack knowledge or competence  
* This is *not* an argument against alignment or safety work  
* This does *not* assume a single mechanism  
* This does *not* require interpreting internal representations  
* This does *not* claim generalisation to all user-model bias  
* This does *not* frame the behaviour as misuse  
* This is *not* a proposal for a specific fix or policy

---

# **7\. Why This Exists Before Anything Else**

I’m writing this now to fix the shape of the problem before committing effort to experiments or tooling. The failure mode I’m interested in is easy to blur into adjacent issues once you start taking measurements, and I want a stable reference point that makes clear what I think is happening, what I think is *not* happening, and what would actually count as progress. Working at this level lets me separate signal from noise, constrain scope, and avoid prematurely optimising for results that answer the wrong question. This document is a way of keeping the work honest, focused, and proportionate before I start pulling on the thread with mechanistic tools.

This document is intended to live in the project workspace as a persistent reference and guiding context for any agentic development associated with this work. It encodes not only the problem framing, but also the **communication constraints** under which **productive work** actually occurs for me:

* Direct, reasoning-first responses  
* Minimal verbosity  
* No performative agreement/sycophancy  
* No unnecessary formalism  
* Local, proportional adjustments rather than global reframing in response to feedback.

The goal is to avoid **repeatedly re-establishing** these constraints or retraining an agent through **corrective interaction**. 

*Any agent operating in this workspace should treat this document as fixed context and default to advancing the work within its bounds, rather than renegotiating tone, abstraction level, or intent.*