# Datasets

This directory stores roleplay training data (not Python code).

## Canonical JSONL format

Each line is a JSON object:

```json
{
  "character": "wizard",
  "messages": [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am Eldrin, guardian of the arcane forest."}
  ],
  "meta": {"source": "local"}
}
```

Add incremental datasets under `datasets/local/`.

