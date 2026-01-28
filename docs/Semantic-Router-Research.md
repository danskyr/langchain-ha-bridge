# Semantic Router Research

Research into the `semantic_router` Python library by Aurelio AI, for evaluating whether to use it for intent classification in our routing layer.

## Core Mechanism

Semantic Router is an embedding-based classification system that routes queries **without using an LLM for the routing decision**. Under the hood it is **kNN over utterance embeddings**:

1. At initialization, all utterances (example phrases per route) are encoded into dense embeddings and stored in an index.
2. At query time, the user input is embedded with the same encoder.
3. The index returns the `top_k` (default: 5) most similar utterance embeddings.
4. Scores are grouped by their parent route name.
5. An aggregation function (`mean`, `sum`, or `max`) reduces each route's scores to a single number.
6. The highest-scoring route is checked against a `score_threshold`. If it passes, the route is returned. Otherwise `None` is returned (no match).

Similarity metric is **cosine similarity** (dot product of normalized vectors).

## Route Definitions and Utterances

A `Route` is a Pydantic model with:

- `name` - identifier for the route
- `utterances` - list of example phrases that define the route's semantic space
- `score_threshold` - optional per-route threshold (overrides global)
- `function_schemas` - for dynamic routes (see below)

Utterances are not templates or regex patterns. They are sample inputs that define what kind of query should trigger the route. The more diverse and representative the utterances, the better the coverage. Each utterance is individually embedded and stored in the index, tagged with its parent route name.

```python
from semantic_router import Route

iot_control = Route(
    name="iot_control",
    utterances=[
        "turn on the living room lights",
        "set the thermostat to 72",
        "lock the front door",
    ],
    score_threshold=0.5,
)
```

## Encoders

22 encoder integrations are supported:

| Category | Encoders |
|---|---|
| Cloud APIs | OpenAI, Azure OpenAI, Cohere, Google, Mistral, Jina, Voyage, NVIDIA NIM, AWS Bedrock, LiteLLM |
| Local/Open-source | HuggingFace (`all-MiniLM-L6-v2` default), FastEmbed, Ollama |
| Sparse/Keyword | BM25, TF-IDF |
| Vision/Multi-modal | CLIP, ViT |
| Hybrid | Aurelio (dense + sparse) |

Supports asymmetric encoding where query embeddings differ from document/utterance embeddings. Also supports hybrid search (dense + sparse) via an `alpha` parameter (0 = pure dense, 1 = pure sparse, default 0.3).

## Decision Flow Detail

```
__call__(text) ->
  1. _encode(text, input_type="queries")
  2. index.query(embedding, top_k=5)
  3. group_scores_by_class(results)        # {route_name: [score1, score2, ...]}
  4. _score_routes(grouped_scores)          # aggregate per-route
  5. _pass_routes(scored_routes)            # filter by threshold
  6. return RouteChoice(name, function_call, similarity_score)
```

Key parameters on `BaseRouter`:

| Parameter | Default | Purpose |
|---|---|---|
| `top_k` | 5 | Number of nearest utterances retrieved |
| `aggregation` | `"mean"` | How to combine utterance scores per route |
| `score_threshold` | `None` | Global threshold (per-route overrides take precedence) |
| `auto_sync` | `None` | Sync mode for remote indexes |

Score aggregation example: if a query matches 3 utterances from a route with cosine similarities [0.85, 0.72, 0.68], the score with `mean` is 0.75, with `max` is 0.85.

## Dynamic Routes

Dynamic routes add LLM-based parameter extraction **on top of** the vector similarity routing. The routing decision is still vector-based, but once a route is selected, an LLM extracts structured parameters.

1. Vector similarity selects the route (same as static).
2. The LLM is invoked with the user query and function schemas.
3. The LLM returns structured JSON with function name and extracted arguments.

A single route can have multiple function schemas. The LLM decides which are relevant. This reintroduces LLM latency for those specific routes only.

## Index Backends

| Index | Storage | Persistence | Best For |
|---|---|---|---|
| `LocalIndex` | In-memory numpy arrays | Ephemeral | Dev, small route sets |
| `HybridLocalIndex` | In-memory (dense + sparse) | Ephemeral | Hybrid search dev |
| `PineconeIndex` | Pinecone cloud | Persistent | Production, large scale |
| `QdrantIndex` | Qdrant cloud/self-hosted | Persistent | Production, self-hosted |

`LocalIndex` is brute-force cosine similarity over numpy arrays. Adequate for up to a few thousand utterances.

`auto_sync` controls how local route definitions and remote index state stay in sync:
- `"local"`: local routes are source of truth
- `"remote"`: remote index is source of truth
- `None`: no automatic sync

## Performance Characteristics

| Aspect | Detail |
|---|---|
| Routing decision (excluding encoding) | Sub-millisecond for LocalIndex |
| Encoding latency (cloud) | ~20-100ms |
| Encoding latency (local) | ~5-20ms |
| Total end-to-end | ~100ms typical |
| LLM-based routing comparison | ~5000ms |
| Cold start (local encoder) | Several seconds for model load |
| Init cost | All utterances encoded at construction time |

## Threshold Tuning

The library provides a `fit()` method that performs random search over thresholds using labeled (utterance, expected_route) pairs. The `evaluate()` method computes classification accuracy. This requires manually curated labeled data.

## Limitations and Trade-offs

1. **Utterance quality is critical.** The system is only as good as the example phrases. Writing good utterances is similar to writing good few-shot examples.

2. **No compositional reasoning.** Selects exactly one route or none. Cannot combine or span multiple routes for a single query.

3. **Threshold sensitivity.** A single score boundary per route determines match vs no-match. Too low causes false positives; too high causes false negatives.

4. **Semantic overlap degrades accuracy.** When routes have similar semantic spaces (e.g., "weather info" vs "climate discussion"), the router struggles to discriminate. Embeddings are not route-aware.

5. **No online learning.** Does not improve from misrouted queries in production. Utterances and thresholds must be manually adjusted.

6. **Encoder lock-in.** Switching encoders requires re-encoding everything. Embeddings from different models are incompatible.

7. **Scaling accuracy.** Computes fine with many routes, but embedding space gets crowded and discrimination drops as route count grows.

8. **Dynamic routes reintroduce LLM latency.** The speed advantage is partially lost when parameter extraction is needed.

## Relevance to Our System

Our current LangGraph router agent (`router_agent_v2.py`) handles multi-round tool calling, state management, and parallel handler execution. Semantic Router would only replace the **initial routing decision** (IOT vs Search vs General), not the handler execution.

With only 3 well-separated intents, the benefit over the current approach is marginal unless routing latency is a bottleneck. The real value shows up when you have many routes or need sub-100ms classification decisions.

If we adopt it, it would slot in as the classification step before dispatching to handlers, while LangGraph continues to manage state and tool execution downstream.
