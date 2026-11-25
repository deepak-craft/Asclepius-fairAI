from pathlib import Path
import base64

BASE_DIR = Path(__file__).resolve().parents[2]


def load_page_template(template_name: str) -> str:

    template_path = (
        BASE_DIR
        / "templates"
        / "components"
        / template_name
    )

    return template_path.read_text(
        encoding="utf-8"
    )


def load_asset(asset_type: str, asset_name: str) -> str:

    asset_path = (
        BASE_DIR
        / "assets"
        / asset_type
        / asset_name
    )

    return asset_path.read_text(
        encoding="utf-8"
    )


def load_image(image_name: str) -> str:

    image_path = (
        BASE_DIR
        / "assets"
        / "images"
        / image_name
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    suffix = image_path.suffix.lower().replace(".", "")
    mime_type = "image/png" if suffix in ["png", ""] else f"image/{suffix}"
    if mime_type == "image/jpg":
        mime_type = "image/jpeg"

    return f"data:{mime_type};base64,{encoded}"