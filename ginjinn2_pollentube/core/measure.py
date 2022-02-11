import multiprocessing
import os
import warnings
import xml.etree.ElementTree as et
from typing import Dict, Generator, Tuple, Union

import astropy.units as u
import fil_finder
import numpy as np
import pandas as pd
import skimage
import skimage.draw
import skimage.morphology
import tqdm

from ginjinn.utils.utils import load_coco_ann, get_obj_anns


def get_local_segmentation_mask(obj_ann) -> np.ndarray:
    '''get_local_segmentation_mask

    Parameters
    ----------
    obj_ann : dict
        Object annotation in COCO format

    Returns
    -------
    np.ndarray
        Local segmentation mask
    '''
    bbox = obj_ann['bbox']
    seg = np.array(obj_ann['segmentation'][0]).reshape(-1, 2)

    x_min, y_min, w, h = bbox

    seg_local = seg - np.array([x_min, y_min])

    return skimage.draw.polygon2mask((h, w), seg_local[:, [1, 0]])


def generate_measurement_input(
    ann_file: str, scale: Union[Dict[str, float], float],
) -> Generator[Tuple[Dict, str, float], None, None]:
    '''generate_measurement_input

    Parameters
    ----------
    ann_file : str
        Path to annotation file in COCO format (JSON)
    scale : float or dict
        Scale in millimeters per pixel. If float, a constant scale across all images will
        be assumed. A dict (key = image file name, value = mm per px) can be used to account
        for different scales across images.

    Yields
    -------
    tuple
        (obj_ann, img_name, scale)
        Here scale is a float referring to the current image.
    '''
    ann = load_coco_ann(ann_file)

    for img_ann in ann['images']:
        img_name = img_ann['file_name']

        if isinstance(scale, dict):
            mm_per_px = scale[img_name]
        else:
            mm_per_px = scale

        for obj_ann in get_obj_anns(img_ann, ann):
            if len(obj_ann['segmentation']) > 1:
                print(
                    f'Skip non-contiguous instance in {img_name} (more than one polygon found).'
                )
            else:
                yield (obj_ann, img_name, mm_per_px)


def count_valid_instances(ann_file: str,) -> int:
    '''count_valid_instances

    Parameters
    ----------
    ann_file : str
        Path to annotation file in COCO format (JSON)

    Returns
    -------
    float
        Number of objects characterized by a single polygon (necessarily contiguous)
    '''
    ann = load_coco_ann(ann_file)
    i_valid = 0

    for img_ann in ann['images']:
        for obj_ann in get_obj_anns(img_ann, ann):
            if len(obj_ann['segmentation']) == 1:
                i_valid += 1

    return i_valid


def measure_skeleton(obj_ann: Dict, scale: float) -> float:
    '''measure_skeleton

    Measure single segmented pollen tube.

    Parameters
    ----------
    obj_ann : dict
        Object annotation in COCO format
    scale : float
        Scale in millimeters per pixel

    Returns
    -------
    float
        tube_length
    '''
    seg_mask = get_local_segmentation_mask(obj_ann)
    skeleton = skimage.morphology.skeletonize(seg_mask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fil = fil_finder.FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
        fil.medskel(verbose=False)
        fil.analyze_skeletons(skel_thresh=10 * u.pix, prune_criteria='length')

    tube_length_px = fil.lengths().value[0]
    tube_length_mm = tube_length_px * scale
    return tube_length_mm


def measure_skeleton_wrapper(
    input_tuple: Tuple[Dict, str, float],
) -> Tuple[str, Union[float, IndexError]]:
    '''measure_skeleton_wrapper

    Wraps measure_skeleton() for parallel execution

    Parameters
    ----------
    input_tuple : Tuple[Dict, str, float]
        (obj_ann, img_name, mm_per_px) as yielded by generate_measurement_input()

    Returns
    -------
    Tuple[str, Union[float, IndexError]]
        (img_name, tube_length) or (img_name, error)
    '''
    obj_ann, img_name, mm_per_px = input_tuple
    try:
        tube_length = measure_skeleton(obj_ann, mm_per_px)
    except IndexError as err:
        return (img_name, err)
    return (img_name, tube_length)


def measure_tube_lengths_from_seg(
    ann_file: str,
    scale: Union[Dict[str, float], float],
    correction: Tuple[float, float] = (0, 1),
    n_proc=None,
) -> pd.DataFrame:
    '''measure_tube_lengths_from_seg

    Parameters
    ----------
    ann_file : str
        Path to annotation file in COCO format (JSON)
    scale : float or dict
        Scale in millimeters per pixel. If float, a constant scale across all images will
        be assumed. A dict (key = image file name, value = mm per px) can be used to account
        for different scales across images.
    correction : tuple of float
        A tuple of length two: (y-intercept, slope).
        This allows to correct for a linear bias of the length measurements.
    n_proc : int
        Number of worker processes, by default os.cpu_count()

    Returns
    -------
    pd.DataFrame
        (Corrected) length measurements of individual pollen tubes
    '''
    with multiprocessing.Pool(n_proc) as p:
        tube_data = [
            data
            for data in tqdm.tqdm(
                p.imap(
                    measure_skeleton_wrapper,
                    generate_measurement_input(ann_file, scale),
                ),
                total=count_valid_instances(ann_file),
            )
            if isinstance(data[1], float)  # TODO: better error handling?
        ]

    tube_df = pd.DataFrame(tube_data, columns=['image_name', 'length_mm'])
    tube_df['length_mm'] = correction[0] + correction[1] * tube_df['length_mm']

    return tube_df


def get_scales_from_polylines(
    ann_file: str, scale_label: str = 'Scale'
) -> Union[Dict[str, float], float]:
    '''
    Get size scale from CVAT-Image XML annotation.

    Parameters
    ----------
    ann_file : str
        Annotation file in CVAT-Image format (XML).
    scale_label: str
        Category name of the scale polyline(s).

    Returns
    -------
    Union[Dict[str, float], float]
        Either a single float determining the scale in mm per pixel for all images if
        only one scale is present in the dataset or alternatively a Dictionary mapping
        image names to scales in mm per pixel.

    Raises
    ------
    Exception
        Raised if an invalid amount of scales is found in the dataset.
    '''
    tree = et.parse(ann_file)
    root = tree.getroot()

    n_images = 0
    for _ in root.findall('image'):
        n_images += 1

    if n_images == 0:
        raise Exception('Could not find any image annotations in annotation file')

    scale_nodes = list(root.findall(f'image/polyline[@label="{scale_label}"]'))
    n_scales = len(scale_nodes)

    if (n_scales != 1) and (n_scales != n_images):
        msg = (
            'Expected either one single scale annotation for the whole dataset '
            f'or one scale annotation for each image, but found {n_scales} scale annotations.'
        )
        raise Exception(msg)

    scales = {}
    for image_node in root.findall('image'):
        image_name = image_node.attrib['name']

        scale_node = image_node.find(f'polyline[@label="{scale_label}"]')
        if scale_node is None:
            continue

        xy: np.ndarray = np.array(
            [p.split(',') for p in scale_node.attrib['points'].split(';')], dtype=float
        )[[0, -1]]
        if len(xy) > 2:
            msg = (
                'Encountered scale with more than 2 points in annotation of image {image_name}. '
                'Only the first two points will be used.'
            )
            warnings.warn(msg)

        length_px: float = np.sqrt(np.sum(np.power(xy[0] - xy[1], 2)))
        scale_node = scale_node.find('attribute[@name="Length"]')
        if scale_node is None:
            msg = f'Could not find scale in annotation of image {image_name}.'
            raise Exception(msg)
        scale_str = scale_node.text
        if scale_str is None:
            msg = f'Text field of scale in annotation of {image_name} is empty.'
            raise Exception(msg)
        length_mm = float(scale_str)
        mm_per_px = length_mm / length_px

        scales[image_name] = mm_per_px

    if n_scales == 1:
        return next(iter(scales.values()))

    return scales


def calc_tube_length(xy: np.ndarray) -> float:
    '''calc_tube_length

    Parameters
    ----------
    xy : np.ndarray
        Tube coordinates

    Returns
    -------
    float
        Tube length
    '''
    return np.sum(np.sqrt(np.sum(np.power(xy[0:-1] - xy[1:], 2), axis=1)))


def measure_tube_lengths_from_polyline(
    ann_file: str, scale: Union[Dict[str, float], float]
) -> pd.DataFrame:
    '''measure_tube_lengths_from_polyline

    Parameters
    ----------
    ann_file : str
        Path to annotation file in CVAT format (XML)
    scale : float or dict
        Scale in millimeters per pixel. If float, a constant scale across all images will
        be assumed. A dict (key = image file name, value = mm per px) can be used to account
        for different scales across images.

    Returns
    -------
    pd.DataFrame
        Average tube length per image/group.
    '''
    tree = et.parse(ann_file)
    root = tree.getroot()

    images = root.findall('image')
    tube_data = []

    for image in images:
        image_name = image.attrib['name']

        if isinstance(scale, dict):
            mm_per_px = scale[image_name]
        else:
            mm_per_px = scale

        for i, pollen_tube in enumerate(
            image.findall('polyline[@label="Pollen_tube"]')
        ):
            tube_name = f'{os.path.splitext(image_name)[0]}_{i}'

            points_str = pollen_tube.attrib['points']
            xy = np.array([s.split(',') for s in points_str.split(';')]).astype('float')

            tube_length_px = calc_tube_length(xy)
            tube_length_mm = tube_length_px * mm_per_px

            tube_data.append([tube_name, image_name, tube_length_px, tube_length_mm])

    tube_df = pd.DataFrame(
        tube_data, columns=['tube_name', 'image_name', 'length_px', 'length_mm']
    )

    return tube_df
