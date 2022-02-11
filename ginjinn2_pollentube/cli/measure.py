from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from pathlib import Path
import cloup
import click
from cloup import (
    command,
    option,
    help_option,
    Path as PathType,
    option_group,
)

if TYPE_CHECKING:
    import pandas as pd


def measure_tubes_polylines(
    ann_file: str, scales: Union[None, Dict[str, float], float] = None
) -> 'pd.DataFrame':
    from ..core.measure import (
        measure_tube_lengths_from_polyline,
        get_scales_from_polylines,
    )

    scale = get_scales_from_polylines(ann_file) if scales is None else scales

    tube_df = measure_tube_lengths_from_polyline(ann_file=ann_file, scale=scale)

    return tube_df


def measure_tubes_segmentations(
    ann_file: str,
    scales: Union[str, Dict[str, float], float],
    correction: Tuple[float, float] = (0.0, 1.0),
) -> 'pd.DataFrame':
    from ..core.measure import (
        measure_tube_lengths_from_seg,
        get_scales_from_polylines,
    )

    if isinstance(scales, str):
        scales = get_scales_from_polylines(scales)

    tube_df = measure_tube_lengths_from_seg(
        ann_file=ann_file, scale=scales, correction=correction,
    )

    return tube_df


class FloatOrPath(cloup.ParamType):
    name = "Path or Float"

    path_type = PathType(exists=True, file_okay=True, dir_okay=False, path_type=Path)

    def convert(self, value, param, ctx):
        if isinstance(value, Path):
            return value

        try:
            value = float(value)
            return value
        except ValueError as val_err:
            try:
                value = self.path_type.convert(value, param, ctx)
                return value.decode()
            except click.BadParameter as err:
                self.fail(
                    (
                        f'{value!r} is neither a valid float ({val_err}) nor a valid file path '
                        f'({err}).'
                    ),
                    param,
                    ctx,
                )


class AnnotationPathType(PathType):
    name = "Annotation File"

    def __init__(self):
        super().__init__(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=False,
            allow_dash=False,
            path_type=Path,
        )

    def convert(self, value, param, ctx):
        ann_file: Path = Path(super().convert(value, param, ctx).decode())

        ann_ext = ann_file.suffix
        if ann_ext.lower() not in ('.json', '.xml'):
            self.fail(
                (f'{ann_file!r} is neither an XML nor a JSON file.'), param, ctx,
            )

        return ann_file


def get_ann_type_from_path(ann_file: Path) -> str:
    ext = ann_file.suffix
    if ext.lower() == '.json':
        return 'coco'
    elif ext.lower() == '.xml':
        return 'cvat'
    raise Exception(f'Invalid file type "{ext}"')


@command('measure', help='Measure pollen tube length.')
@option(
    '-i',
    '--ann_file',
    type=AnnotationPathType(),
    default=None,
    required=True,
    help='''
        Annotation file in CVAT Image (XML) or COCO (JSON) format. If annotation is
        in COCO format, the -s/--scale is required.
    ''',
)
@option(
    '-o',
    '--out_prefix',
    type=PathType(path_type=str),
    default=None,
    required=True,
    help='''
        Prefix of the output file(s) optionally including path.
    ''',
)
@option(
    '-s',
    '--scale',
    type=FloatOrPath(),
    default=None,
    help='''
        Either an CVAT Image (XML) annotation file with scale annotations
        or a single value determining the scale in mm per pixel. Required if
        -i/--ann_file is a COCO annotation file.
    ''',
)
@option(
    '-g',
    '--group_file',
    type=PathType(
        path_type=str, exists=True, dir_okay=False, readable=True, file_okay=True
    ),
    default=None,
    help='''
        Optional grouping file.
    ''',
)
@option(
    '-f',
    '--force',
    is_flag=True,
    help='''
        Force overwrite existing outputs.
    ''',
)
@help_option('-h', '--help')
def measure(
    ann_file: Path,
    out_prefix: str,
    scale: Union[str, float, None],
    group_file: Optional[str],
    force: bool,
):
    import sys
    import pandas as pd

    # check if outputs already exist
    out_tubes = Path(f'{out_prefix}_all.csv')
    out_imgwise = Path(f'{out_prefix}_imgwise.csv')
    out_grpwise = Path(f'{out_prefix}_grpwise.csv')

    if not force:
        if out_tubes.exists():
            sys.stderr.write(
                f'Error: file {out_tubes} already exists. Rerun with -f/--force to overwrite.\n'
            )
            sys.exit(1)
        if out_imgwise.exists():
            sys.stderr.write(
                f'Error: file {out_imgwise} already exists. Rerun with -f/--force to overwrite.\n'
            )
            sys.exit(1)
        if not group_file is None and out_grpwise.exists():
            sys.stderr.write(
                f'Error: file {out_grpwise} already exists. Rerun with -f/--force to overwrite.\n'
            )
            sys.exit(1)

    # measure tubes
    ann_type = get_ann_type_from_path(ann_file)
    if ann_type == 'coco':
        if scale is None:
            msg = (
                'Error: \'-s\'/\'--scale\' is required if \'-i\'/\'--ann_file\' '
                'is a COCO (JSON) annotation file.\n'
            )
            sys.stderr.write(msg)
            sys.exit(1)

        tube_df = measure_tubes_segmentations(
            ann_file=str(ann_file), scales=scale, correction=(0, 1)
        )
    else:
        tube_df = measure_tubes_polylines(ann_file=str(ann_file), scales=scale)

    tube_df.to_csv(out_tubes, index=False)
    print(f'Tube-wise measurements written to {out_tubes}.')

    tube_df_imgwise = tube_df.groupby('image_name')[
        ['image_name', 'length_mm']
    ].describe()
    tube_df_imgwise = tube_df_imgwise['length_mm']
    tube_df_imgwise['count'] = pd.to_numeric(
        tube_df_imgwise['count'], downcast='integer'
    )
    tube_df_imgwise.to_csv(out_imgwise, index=True)
    print(f'Image-wise measurements written to {out_imgwise}.')

    if not group_file is None:
        group_df = pd.read_csv(group_file, dtype=str, sep='\s+', header=None)
        group_dict = group_df.set_index(0).to_dict()[1]
        tube_df['image_group'] = [group_dict[img] for img in tube_df['image_name']]
        tube_df_groupwise = tube_df.groupby('image_group')[
            ['image_group', 'length_mm']
        ].describe()
        tube_df_groupwise = tube_df_groupwise['length_mm']
        tube_df_groupwise['count'] = pd.to_numeric(
            tube_df_groupwise['count'], downcast='integer'
        )
        tube_df_groupwise.to_csv(out_grpwise, index=True)
        print(f'Group-wise measurements written to {out_grpwise}.')
