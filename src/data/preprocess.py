"""Page preprocessing shared by the working-copy build and inference.

Decision (PRD 5.1.3): **plain resize** to a square, ignoring the aspect ratio.
    * It is what the RVL-CDIP literature does (Harley et al. 2015 resize whole
      pages), so our numbers stay comparable.
    * Pad-to-square would shrink a tall page to roughly 75% of the width,
      wasting pixels on padding and making small text and table lines thinner.
The same function is used for RVL-CDIP and for PDF pages, so every model sees
identically prepared input.
"""

from __future__ import annotations

from PIL import Image

WORKING_COPY_SIZE = 256  # stored size; models then resize to data.image_size (224)


def to_working_copy(image: Image.Image, size: int = WORKING_COPY_SIZE) -> Image.Image:
    """Return the first page of ``image`` as a ``size`` x ``size`` 8-bit grayscale image."""
    if getattr(image, "n_frames", 1) > 1:
        image.seek(0)  # multi-page TIFF: RVL-CDIP labels describe the first page
    # Convert before resizing: many RVL-CDIP scans are 1-bit fax images, and
    # resizing in 8-bit grayscale gives anti-aliased strokes instead of jaggies.
    gray = image.convert("L")
    return gray.resize((size, size), Image.Resampling.LANCZOS)
