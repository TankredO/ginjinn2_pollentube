from typing import Tuple
from cloup import command, option, help_option, Path as PathType


@command('convert', help='Convert CVAT Image (XML) dataset to COCO dataset.')
@option(
    '-i',
    '--ann_file',
    type=PathType(dir_okay=False, path_type=str),
    default=None,
    required=True,
    help='''
        Annotation file in CVAT Image (XML) format.
    ''',
)
@option(
    '-o',
    '--out_file',
    type=PathType(dir_okay=False, path_type=str),
    default=None,
    required=True,
    help='''
        COCO annotation (JSON) output file.
    ''',
)
@option(
    '-b',
    '--buffer_size',
    multiple=True,
    type=str,
    help='''
        Buffer sizes (expansion radius) for converting polylines to polygons in pixels, 
        passed as a pair of category name (as annotated in CVAT) and
        numeric value. E.g., "PollenTube=3" if the pollen tube category
        is "PollenTube" and the expected width of the pollen tubes is 6px.
    ''',
    default=('Pollen_tube=5',),
    show_default=True,
)
@option(
    '-x',
    '--exclude',
    multiple=True,
    type=str,
    help='''
        Category names to exclude from measuring, e.g. scale categories.
    ''',
    default=('Scale',),
    show_default=True,
)
@option(
    '-f',
    '--force',
    is_flag=True,
    help='''
        Force overwrite existing output.
    ''',
)
@help_option('-h', '--help')
def convert(
    ann_file: str,
    out_file: str,
    buffer_size: Tuple[str, ...],
    exclude: Tuple[str, ...],
    force: bool,
):
    buffer_sizes = {}
    for buf_str in buffer_size:
        cat_name, _, buffer = buf_str.partition('=')
        buffer_sizes[cat_name] = float(buffer)

    import json
    import sys
    from pathlib import Path
    from ..core import conversion

    out_path = Path(out_file)
    if out_path.exists() and not force:
        sys.stderr.write(
            f'Error: file {out_path} already exists. Rerun with -f/--force to overwrite.\n'
        )
        sys.exit(1)

    coco_ds = conversion.polyline_xml_to_polygon_coco(
        ann_file=ann_file, buffer_sizes=buffer_sizes, exclude_labels=exclude
    )

    with open(out_path, 'w') as out_f:
        json.dump(coco_ds, out_f)

    print(f'COCO dataset written to {out_path}.')
