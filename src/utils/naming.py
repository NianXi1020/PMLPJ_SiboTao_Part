"""Filename and naming helpers."""


def safe_model_name(model_name: str) -> str:
    """Normalize model names for filenames."""
    return model_name.lower().replace("-", "").replace(" ", "_")


def figure_name(dataset: str, model: str, plot_name: str) -> str:
    """Build standardized PNG filename."""
    return f"{dataset}_{safe_model_name(model)}_{plot_name}.png"
