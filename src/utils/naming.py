"""Filename and naming helpers."""


def safe_model_name(model_name: str) -> str:
    """Normalize model names for filenames."""
    return model_name.lower().replace("-", "").replace(" ", "_")


def artifact_name(
    dataset: str,
    model_or_common: str,
    output_name: str,
    timestamp: str,
    extension: str,
) -> str:
    """Build standardized artifact filename with run timestamp."""
    return f"{dataset}_{safe_model_name(model_or_common)}_{output_name}_{timestamp}.{extension}"


def figure_name(dataset: str, model_or_common: str, plot_name: str, timestamp: str) -> str:
    """Build standardized PNG filename with run timestamp."""
    return artifact_name(
        dataset=dataset,
        model_or_common=model_or_common,
        output_name=plot_name,
        timestamp=timestamp,
        extension="png",
    )


def table_name(dataset: str, model_or_common: str, table_type: str, timestamp: str) -> str:
    """Build standardized CSV filename with run timestamp."""
    return artifact_name(
        dataset=dataset,
        model_or_common=model_or_common,
        output_name=table_type,
        timestamp=timestamp,
        extension="csv",
    )
