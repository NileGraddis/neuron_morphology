import logging
import copy as cp

import shapely
import numpy as np

from argschema.argschema_parser import ArgSchemaParser

from neuron_morphology.snap_polygons._schemas import (
    InputParameters, OutputParameters)
from neuron_morphology.snap_polygons._from_lims import FromLimsSource
from neuron_morphology.snap_polygons.geometries import (
    Geometries, make_scale, clear_overlaps, closest_from_stack, 
    get_snapped_polys, find_vertical_surfaces
)
from neuron_morphology.snap_polygons.types import ensure_linestring
from neuron_morphology.snap_polygons.image_outputter import ImageOutputter
from neuron_morphology.transforms.geometry import get_vertices_from_two_lines


def trim_to_close(geom, threshold, linestring):
    linestring = ensure_linestring(cp.deepcopy(linestring))

    crit = lambda pt: geom.distance(shapely.geometry.Point(pt)) <= threshold
    coords = trim_coords(linestring.coords, crit)

    return shapely.geometry.LineString(coords)


def find_transition(unmet, met, criterion, max_iter=100):
    """
    """

    segment = shapely.geometry.LineString([unmet, met])
    midpoint = segment.interpolate(segment.length / 2)

    if criterion(midpoint):
        met = midpoint
    else:
        unmet = midpoint

    if max_iter > 0:
        return find_transition(unmet, met, criterion, max_iter-1)
    return met

def first_met(coords, criterion, description=""):
    
    for idx, coord in enumerate(coords):
        coord = shapely.geometry.Point(coord)
        met = criterion(coord)

        if met:
            if idx == 0:
                return idx, None
            return idx, find_transition(coords[idx-1], coord, criterion)

    raise ValueError(f"criterion ({description}) never met!")


def trim_coords(coords, criterion):  
    print([(coord, criterion(coord)) for coord in coords])

    left_index, left_pt = first_met(coords, criterion)
    right_index, right_pt = first_met(coords[::-1], criterion)

    if right_index > 0:
        coords = coords[:-right_index] + [right_pt]
    if left_index > 0:
        coords = [left_pt] + coords[left_index:]

    return coords


def run_snap_polygons(
    layer_polygons, 
    pia_surface, 
    wm_surface, 
    layer_order,
    working_scale: float,
    images=None,
    surface_distance_threshold: float = 400.0,
    multipolygon_error_threshold: float = 10**4
):
    """
    """

    # setup input geometries
    geometries = Geometries()
    geometries.register_polygons(layer_polygons)
    
    # setup cortex boundaries
    hull = geometries.convex_hull()
    pia = trim_to_close(hull, surface_distance_threshold, pia_surface["path"])
    wm = trim_to_close(hull, surface_distance_threshold, wm_surface["path"])
    
    geometries.register_surface("pia", pia)
    geometries.register_surface("wm", wm)

    pia_wm_vertices = get_vertices_from_two_lines(pia.coords[:], wm.coords[:])
    bounds = shapely.geometry.polygon.Polygon(pia_wm_vertices)

    # go!
    result_geos = (
        geometries
        .fill_gaps(working_scale)
        .cut(bounds)
    )

    # get output surfaces
    boundaries = find_vertical_surfaces(
        result_geos.polygons, 
        layer_order, 
        pia=geometries.surfaces["pia"], 
        wm=geometries.surfaces["wm"]
    )
    result_geos.register_surfaces(boundaries)        

    # write results
    outputter = ImageOutputter(
        geometries, result_geos, images
    )
    results = result_geos.to_json()
    results["images"] = outputter.write_images()
    return results


def main():

    class Parser(ArgSchemaParser):
        """
        """
        default_configurable_sources = \
            ArgSchemaParser.default_configurable_sources + [FromLimsSource]

    parser = Parser(
        schema_type=InputParameters,
        output_schema_type=OutputParameters
    )

    args = cp.deepcopy(parser.args)
    logging.getLogger().setLevel(args.pop("log_level"))

    output = run_snap_polygons(**args)
    output.update({"inputs": parser.args})

    parser.output(output)


if __name__ == "__main__":
    main()
