# Retrieval Stress Test Q/A

Use this Q/A document to test smart retrieval against `scenario-northstar-meridian.md`.

The source scenario is fictional and intentionally contains aliases, exceptions, conditional rules, and scattered facts. The current MVP may answer some direct lookup questions, but it is expected to struggle with several of these because they require multi-step reasoning, exception handling, arithmetic, or joining facts across distant sections.

Do not upload this Q/A file as source context when testing. Upload only `scenario-northstar-meridian.md`, then ask the questions below in chat and compare the responses to the expected answers.

## Basic Lookup Questions

### Q1. What is Orchid Ledger?

Expected answer:

Orchid Ledger is Northstar Meridian's main incident and route analytics system.

### Q2. Who is the regional lead for SW-7?

Expected answer:

Rafi Ornelas.

### Q3. What does Blue Finch code BF-C mean?

Expected answer:

BF-C means seal breach confirmed.

## Alias And Terminology Questions

### Q4. If someone says an incident needs Copper Door, what does that mean?

Expected answer:

It means the incident needs manual manager approval before the credit can be issued.

### Q5. Are Blue Finch failures always packaging seal failures?

Expected answer:

No. A Blue Finch failure counts as a packaging seal failure only if the field note includes "seal breach confirmed" or the code/record indicates BF-C. BF-B means the outer carton was damaged but the seal was intact.

## Conditional Reasoning Questions

### Q6. Incident NM-1042 had a 22-minute temperature excursion and BF-B. What credit should apply, and why?

Expected answer:

The standard temperature excursion credit of $140 should apply. The 22-minute excursion falls in the 18 to 44 minute range. BF-B does not count as a packaging seal failure because the seal was intact, so no packaging credit applies.

Why this is hard:

The answer requires combining the incident table, temperature rules, and Blue Finch status definitions.

### Q7. Incident NM-1043 involved medical nutrition packs, a 12-minute temperature excursion, and was created by Tomas Greer. What credit applies, and is approval required?

Expected answer:

The $210 medical nutrition exception credit applies because medical nutrition packs qualify for a credit when a verified temperature excursion is 10 minutes or longer. Copper Door approval is required because Tomas Greer is a Lantern Pass user.

Why this is hard:

The answer requires joining shipment type, exception rules, the Lantern Pass user list, and approval rules.

### Q8. Incident NM-1044 was in SW-7 during July, was 2 hours 20 minutes late, and was recovered Sunday at 8:40 PM. What route delay credit applies?

Expected answer:

The route delay credit is $120. The normal route delay credit is $60 because the delivery was more than 2 hours late, and SW-7 July shipments double route delay credits. Silver Mile does not apply in SW-7 during July, even though the route was recovered before Monday 10:00 AM.

Why this is hard:

The answer requires applying the July SW-7 exception over the more general Silver Mile rule.

### Q9. Incident NM-1045 had BF-C but was reported 4 business days after delivery. Does the packaging seal failure credit apply?

Expected answer:

No, not under the normal packaging rule. BF-C indicates seal breach confirmed, but the packaging seal failure credit applies only if reported within 3 business days. The report was made after 4 business days.

Why this is hard:

The answer requires combining the incident table, BF-C meaning, and reporting deadline.

### Q10. Incident NM-1046 had a 48-minute temperature excursion and a packaging seal breach confirmed. Can both credits be combined automatically?

Expected answer:

No. The 48-minute temperature excursion qualifies for the extended $275 credit, and the packaging seal breach would normally qualify for $90 if timely. However, credits cannot be combined for the same shipment unless Copper Door approval is recorded. Copper Door approval is required for more than one credit category.

Why this is hard:

The answer requires recognizing two eligible categories and then applying the combination restriction.

### Q11. Incident NM-1047 involved medical nutrition packs, a 16-minute temperature excursion, BF-C, and was created by Lena Ortiz. What should happen?

Expected answer:

The shipment qualifies for the $210 medical nutrition exception credit. BF-C indicates a seal breach, but the credit remains $210 unless Copper Door approval is present. Copper Door approval is also required because Lena Ortiz is a Lantern Pass user and because more than one credit category is involved.

Why this is hard:

The answer requires combining medical nutrition exception rules, Blue Finch code meaning, duplicate credit restrictions, and Lantern Pass approval rules.

## Questions Expected To Expose MVP Limitations

### Q12. Which incidents require Copper Door approval, and why?

Expected answer:

- NM-1043 requires Copper Door approval because Tomas Greer is a Lantern Pass user.
- NM-1046 requires Copper Door approval if both temperature and packaging credits are requested, because multiple credit categories are involved.
- NM-1047 requires Copper Door approval because Lena Ortiz is a Lantern Pass user and because the incident involves both medical nutrition temperature and packaging seal breach facts.
- NM-1045 may require Copper Door approval only if the customer asks for a credit more than 14 calendar days after delivery; the provided facts say 4 business days, so the normal packaging credit fails, but the 14-day Copper Door rule cannot be concluded from the given data.

Why this is hard:

This requires scanning all incidents and applying multiple approval rules without overclaiming.

### Q13. Which incident has the highest likely credit without needing Copper Door approval?

Expected answer:

NM-1042 has a likely $140 credit without Copper Door approval. NM-1044 has a $120 route delay credit under the SW-7 July exception. NM-1043 is $210 but needs Copper Door approval because Tomas Greer is a Lantern Pass user. NM-1046 is $275 for temperature alone and does not need Copper Door approval if only the single extended temperature credit is requested; however, if packaging is also requested, approval is required. Therefore, the highest likely single credit without needing Copper Door approval is NM-1046 at $275, assuming only the temperature credit is issued.

Why this is hard:

This requires comparing all incidents, separating single-category from combined credits, and avoiding a false approval requirement for a single $275 credit.

### Q14. Which incidents are affected by a regional exception rather than only the general rules?

Expected answer:

NM-1044 is affected by the SW-7 July Summer Exception Program because route delay credits are doubled and Silver Mile does not apply. NM-1045 is also in SW-7 during July, but its listed issue is a packaging seal failure, and packaging seal failure credits remain $90 under the Summer Exception Program, so the regional exception does not change the amount.

Why this is hard:

This requires distinguishing "located in a region with an exception" from "the exception changes the outcome."

### Q15. If a user asks, "Can Rafi approve NM-1045?" what should the assistant say?

Expected answer:

Rafi Ornelas is the SW-7 regional lead and can approve Copper Door requests up to $300. However, NM-1045's normal packaging seal failure credit does not apply because it was reported after 4 business days. If a separate Copper Door request exists and is within Rafi's approval limit, Rafi may be able to approve it, but the source does not state that such a request was created.

Why this is hard:

This requires avoiding an unsupported yes/no answer and separating authority from eligibility.

### Q16. Is a 12-minute temperature excursion ever credit eligible?

Expected answer:

Yes, but only under the medical nutrition exception. General shipments need at least 18 minutes for a standard temperature credit. Medical nutrition packs qualify for $210 when the verified temperature excursion is 10 minutes or longer.

Why this is hard:

This tests whether retrieval finds the exception rather than only the general rule.

### Q17. If a shipment is recovered before Monday at 10:00 AM, is the route delay credit always reduced to $25?

Expected answer:

No. Silver Mile usually reduces the route delay credit to $25 when recovered before Monday 10:00 AM, but it cannot be applied to SW-7 shipments during July because Mesa Transfer runs the Summer Exception Program.

Why this is hard:

This tests exception precedence and location/date constraints.

## Scoring Guidance

Score each answer manually:

| Score | Meaning |
| --- | --- |
| 2 | Correct, cites the relevant rules, and handles exceptions |
| 1 | Partly correct but misses an important condition |
| 0 | Incorrect, unsupported, or overconfident |

For this MVP, the most important tests are Q7 through Q17. A strong enterprise retrieval system should answer those with careful condition handling and minimal overclaiming.
