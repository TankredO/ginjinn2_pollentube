import xml.etree.ElementTree as et

import warnings
import numpy as np
from shapely.geometry import LineString

from ginjinn.simulation import coco_utils
from ginjinn.utils.utils import bbox_from_polygons

from typing import Dict, Any, Iterable, List


def polyline_xml_to_polygon_coco(
    ann_file: str,
    buffer_sizes: Dict[str, float],
    exclude_labels: Iterable[str] = ('Scale',),
    eps: float = 0.001,
) -> Dict[str, Any]:
    '''
    Create COCO instance segmentation annotations (JSON) from polyline
    annotations in CVAT image (XML) format.

    Parameters
    ----------
    ann_file : str
        Annotation file in CVAT-Image format (XML).
    buffer_sizes : Dict[str, float]
        Dictionary mapping category names to buffer sizes.
    exclude_labels : Iterable[str]
        Polyline categories to exclude.
    eps : float
        Points closer than eps will be merged.

    Returns
    -------
    Dict[str, Any]
        COCO dataset as dictionary.

    Raises
    ------
    KeyError
        Raised if a polyline category without corresponding buffer size
        is encountered.
    '''
    exclude_labels = [*exclude_labels]

    tree = et.parse(ann_file)
    root = tree.getroot()

    n_images = 0
    for _ in root.findall('image'):
        n_images += 1

    if n_images == 0:
        raise Exception('Could not find any image annotations in annotation file')

    # xml to COCO
    cats_coco: Dict[str, Dict[str, Any]] = {}
    anns_coco: List[Dict[str, Any]] = []
    imgs_coco: List[Dict[str, Any]] = []
    licenses_coco = [coco_utils.build_coco_license(0)]
    info_coco = coco_utils.build_coco_info()

    category_id = 1
    image_id = 1
    annotation_id = 1
    for image_node in root.findall('image'):
        image_name = image_node.attrib['name']
        image_width = int(image_node.attrib['width'])
        image_height = int(image_node.attrib['height'])

        image_ann = coco_utils.build_coco_image(
            image_id=image_id,
            file_name=image_name,
            width=image_width,
            height=image_height,
        )

        for polyline_node in image_node.findall('polyline'):
            # get category of polyline
            category_name = polyline_node.attrib['label']
            if category_name in exclude_labels:
                continue

            cat_coco = cats_coco.get(category_name, None)
            if cat_coco is None:
                cat_coco = coco_utils.build_coco_category(
                    category_id=category_id, name=category_name,
                )
                cats_coco[category_name] = cat_coco
                category_id += 1

            # get buffer size for category
            try:
                buffer_size = buffer_sizes[category_name]
            except KeyError:
                msg = (
                    f'Missing buffer size for category "{category_name}". Please provide a buffer size '
                    ' or alternatively exclude the category.'
                )
                raise KeyError(msg) from KeyError

            # convert polyline to polygon by buffering
            points_str = polyline_node.attrib['points']
            xy = np.array([s.split(',') for s in points_str.split(';')]).astype('float')

            poly: np.ndarray = np.array(
                LineString(xy).buffer(buffer_size).boundary.xy
            ).T
            # clamp to image size
            poly[:, 0] = np.clip(  # pylint: disable=unsupported-assignment-operation
                poly[:, 0], 0, image_width  # pylint: disable=unsubscriptable-object
            )
            poly[:, 1] = np.clip(  # pylint: disable=unsupported-assignment-operation
                poly[:, 1], 0, image_height  # pylint: disable=unsubscriptable-object
            )
            # remove duplicate points
            poly = np.array(
                [
                    a
                    for a, b in zip(
                        poly[0:-1,], poly[1:,]  # pylint: disable=unsubscriptable-object
                    )
                    if np.linalg.norm(a - b) > eps
                ]
            )

            seg = [poly.flatten().tolist()]
            bbox = bbox_from_polygons(seg, fmt='xywh').tolist()
            if len(seg[0]) < 6:
                msg = f'Polygon with less than 3 vertices found in annotation for image {image_name}'
                warnings.warn(msg)
                continue

            ann = coco_utils.build_coco_annotation(
                ann_id=int(annotation_id),
                image_id=int(image_id),
                category_id=int(cat_coco['id']),
                bbox=bbox,
                segmentation=seg,
                area=float(bbox[2] * bbox[3]),
            )

            anns_coco.append(ann)

            annotation_id += 1

        imgs_coco.append(image_ann)
        image_id += 1

    ds_coco = coco_utils.build_coco_dataset(
        annotations=anns_coco,
        images=imgs_coco,
        categories=list(cats_coco.values()),
        licenses=licenses_coco,
        info=info_coco,
    )

    return ds_coco
