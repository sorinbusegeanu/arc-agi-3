from __future__ import annotations

from pathlib import Path

from PIL import Image

from v4_5.contracts.bootstrapMediaTypes import BootstrapCaptureBundle, BootstrapPngArtifact
from v4_5.contracts.errors import BootstrapPngExportError
from vlm_v2.frame_writer import _observation_to_image


class PngExporter:
    def export(self, *, bundle: BootstrapCaptureBundle, output_dir: str, scale_factor: int) -> tuple[BootstrapPngArtifact, ...]:
        artifacts = []
        base_dir = Path(output_dir)
        sequence_names = tuple(dict.fromkeys(record.sequence_name for record in bundle.step_records))
        for sequence_name in sequence_names:
            paths = []
            sequence_dir = base_dir / sequence_name / "png"
            sequence_dir.mkdir(parents=True, exist_ok=True)
            sequence_records = [record for record in bundle.step_records if record.sequence_name == sequence_name]
            try:
                for idx, record in enumerate(sequence_records):
                    image = _observation_to_image(record.raw_observation_ref)
                    scaled = image.resize((image.width * int(scale_factor), image.height * int(scale_factor)), resample=Image.Resampling.NEAREST)
                    path = sequence_dir / f"{idx:04d}_{record.action.lower()}.png"
                    scaled.save(path)
                    paths.append(str(path))
            except Exception as exc:
                raise BootstrapPngExportError(str(exc)) from exc
            artifacts.append(BootstrapPngArtifact(schema_version="v4.5", sequence_name=sequence_name, png_paths=tuple(paths), scale_factor=int(scale_factor)))
        return tuple(artifacts)
