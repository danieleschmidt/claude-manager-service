from enum import Enum


class RequestType(Enum):
    SIMPLE = "simple"      # short Q&A -> haiku
    COMPLEX = "complex"    # reasoning, code -> sonnet
    CREATIVE = "creative"  # writing, brainstorm -> sonnet
    ANALYSIS = "analysis"  # data analysis -> opus


class RequestRouter:
    """Route requests to Claude models based on request type and content."""

    MODEL_MAP = {
        RequestType.SIMPLE: "claude-3-haiku-20240307",
        RequestType.COMPLEX: "claude-3-5-sonnet-20241022",
        RequestType.CREATIVE: "claude-3-5-sonnet-20241022",
        RequestType.ANALYSIS: "claude-3-opus-20240229",
    }

    def __init__(self):
        self._custom_routes: dict[str, str] = {}  # keyword -> model

    def classify(self, prompt: str) -> RequestType:
        """Classify a prompt into a RequestType using heuristics."""
        lower = prompt.lower()

        # Check custom routes first
        for keyword in self._custom_routes:
            if keyword.lower() in lower:
                # Return the type corresponding to the custom model, or default to SIMPLE
                return RequestType.SIMPLE  # caller should use route() not classify() for custom

        # Keyword-based classification (order matters: more specific first)
        complex_keywords = {"code", "implement", "debug", "function", "programming", "algorithm"}
        creative_keywords = {"write", "story", "poem", "creative", "essay", "narrative", "brainstorm"}
        analysis_keywords = {"analyze", "analyse", "data", "statistics", "compare", "correlation", "trend"}

        if any(kw in lower for kw in analysis_keywords):
            return RequestType.ANALYSIS
        if any(kw in lower for kw in complex_keywords):
            return RequestType.COMPLEX
        if any(kw in lower for kw in creative_keywords):
            return RequestType.CREATIVE
        if len(prompt) < 100:
            return RequestType.SIMPLE

        return RequestType.SIMPLE

    def route(self, prompt: str) -> str:
        """Route a prompt to a model name."""
        lower = prompt.lower()

        # Check custom routes first
        for keyword, model in self._custom_routes.items():
            if keyword.lower() in lower:
                return model

        req_type = self.classify(prompt)
        return self.MODEL_MAP[req_type]

    def add_custom_route(self, keyword: str, model: str):
        """Add a custom keyword -> model route."""
        self._custom_routes[keyword] = model

    def get_model_for_type(self, req_type: RequestType) -> str:
        """Return the model name for a given request type."""
        return self.MODEL_MAP[req_type]
