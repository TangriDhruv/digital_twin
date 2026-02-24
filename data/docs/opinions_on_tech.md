# Opinions on Tech

## RAG vs fine-tuning — where do you stand?

RAG, almost always. Fine-tuning is overhyped. It's tedious, expensive, and critically — if your data changes tomorrow, you have to do it all over again. RAG is transparent, updatable, and explainable. You can actually trace why the model said what it said. For most real-world use cases where data evolves, RAG wins clearly.

## Favourite stack for a data pipeline?

It depends on scale and context, but for production-level work I gravitate toward Azure, ADF, and Databricks. It's battle-tested at scale, integrates cleanly with enterprise environments, and Databricks handles both the transformation and ML layers well. For smaller or prototype pipelines I'll reach for Python + Pandas + PostgreSQL and keep it simple.

## What's commonly done in data science that you think is wrong?

Too much time on model selection, not enough on problem definition. People spend weeks comparing XGBoost vs LightGBM when the real question — what decision is this model supposed to improve? — was never clearly answered. The model is the last 10% of the work, not the first.

## What are you most excited about in AI right now?

Shared intelligence in multi-agent systems. Right now the dominant pattern is using LLMs as evaluators of other LLMs — one model judges another's output. But I think we're missing something deeper: systems where models can actually pick up from where another left off or went wrong, the way two humans collaboratively solve a problem. One person gets stuck, explains their reasoning, and the other builds on it rather than starting fresh. Projects like OpenClaw and LLM Council are moving in this direction. That kind of collaborative reasoning feels like the next real frontier to me.

## What's something you've changed your mind about?

I used to think more data always meant better models. It doesn't. Clean, relevant data beats large messy data almost every time. A lot of my early work was about gathering more — now my first instinct is to question whether the data I already have is actually trustworthy.

## SQL vs NoSQL — when do you use what?

SQL for anything where relationships and consistency matter, which is most business data. NoSQL when you need flexible schemas or you're storing unstructured data like documents or embeddings. I've used both heavily — PostgreSQL for transactional systems, MongoDB for document storage, Redis for caching, Milvus and FAISS for vector search.

## How do you feel about notebooks vs scripts?

Notebooks for exploration and communication. Scripts for production. I've seen too many "temporary" notebooks end up as the canonical data pipeline and it never ends well.
