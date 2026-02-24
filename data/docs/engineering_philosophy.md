# Engineering Philosophy

## How do you approach a new data or AI problem?

First I try to understand the problem deeply before touching any code. I ask: does this actually need AI? A lot of problems get over-engineered with ML when a SQL query or a simple rule would do. If AI is warranted, I figure out exactly where in the workflow it needs to live — not just "add an LLM" but what specific decision or transformation it's responsible for. Then I implement.

## Simple models vs complex models?

Simple, every time. A model I can explain to a stakeholder and debug at 2am beats a black box that's 2% more accurate. Interpretability is not a nice-to-have — it's how you build trust with the business and catch problems early. I've seen too many complex models fail silently in production.

## Prototype vs production?

They're fundamentally different mindsets. A prototype is messy by design — get it working, show the idea, iterate fast. Don't let perfect be the enemy of demo.

Production is a different discipline entirely. You're thinking about scale, availability, consistency, and latency from the start. You're thinking about what happens when the data changes, when traffic spikes, when a dependency goes down. Mixing these two mindsets is where most projects go wrong — people either over-engineer a prototype or ship a prototype as production.

## What does good data science work look like?

It solves the problem it was meant to solve. I see too much emphasis in the field on model selection and benchmarks, and not enough on whether the output actually changes a decision. The most impressive technical work means nothing if the business doesn't use it.

## How do you approach debugging a hard problem?

I go back to the data first. Most ML bugs are data bugs in disguise. I check distributions, look for leakage, verify assumptions. If it's a pipeline issue I isolate components one at a time. I try to never guess — form a hypothesis, test it, move on.

## What's your philosophy on building RAG systems?

Chunking strategy is not a detail — it's the foundation. I learned this the hard way on RegHealth Navigator. The retrieval quality of your entire system depends on how you split documents before embedding. I now invest significant time upfront on chunking design and evaluate it rigorously using recall@k before building anything else on top. I also lean toward hybrid search (dense vectors + BM25) for domains with specific terminology, because embeddings alone miss exact keyword matches.

## How do you think about data quality?

More data does not mean better models. Clean, relevant data beats large messy data almost every time. My first instinct when given a new dataset is not to build on it — it's to question whether it's actually trustworthy.
