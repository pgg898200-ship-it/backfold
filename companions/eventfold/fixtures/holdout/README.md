# Independent holdout

We froze six additional source files from three independent SDK and client projects before checking codec results on them. These are publicly distributed client test fixtures. We included all eight streams they contained, including error responses at the SSE event level. The holdout comprises 215 frames and 45,109 bytes.

| Upstream | Selected source files | Output streams |
| --- | --- | ---: |
| simonw/llm | Responses basic streaming and tool-use streaming cassettes | 3 |
| pydantic/pydantic-ai | Mistral parallel tool calls with mistral-small-latest, Anthropic code execution, and OpenRouter stream error cassettes | 4 |
| langchain-ai/langchainjs | Claude chat mock SSE | 1 |

We copied the LangChain file verbatim. For the cassettes, we parsed YAML and extracted the response body from every HTTP 200 response with a `text/event-stream` content type. String bodies were encoded as UTF-8. Each recorded HTTP response yields its own stream, preserving its original SSE field ordering, spacing, blank lines, comments, and final bytes. The files contain response bodies only. Local filenames identify the source and the one-based interaction ordinal.

The [manifest](manifest.json) pins the source commit, Git blob SHA-1 and byte count, source URL, interaction ordinal, and final fixture SHA-256. No fixture was filtered based on a benchmark result.

The [simonw/llm license](licenses/simonw-llm-LICENSE) is Apache 2.0. [Pydantic AI](licenses/pydantic-pydantic-ai-LICENSE) and [LangChain JS](licenses/langchain-ai-langchainjs-LICENSE) are MIT licensed. The bundled license files retain upstream notices.
