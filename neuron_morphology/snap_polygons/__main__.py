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
    images=None
):
    """
    """

    geometries = Geometries()
    geometries.register_polygons(layer_polygons)
    
    union = None
    for key, poly in geometries.polygons.items():
        if union is None:
            union = poly
        else:
            # Why convex hull? Sometimes layer drawings are not simple
            # polygons - the corners are actually loops and the coordinates 
            # self-intersect. Since for now we are just interested in trimming 
            # distant surface points, this is good enough
            ls = shapely.geometry.LineString(poly.exterior)
            union = poly.convex_hull.union(union.convex_hull)

    threshold = 400
    pia = trim_to_close(union, threshold, pia_surface["path"])
    wm = trim_to_close(union, threshold, wm_surface["path"])
    
    geometries.register_surface("pia", pia)
    geometries.register_surface("wm", wm)

    pia_wm_vertices = get_vertices_from_two_lines(pia.coords[:], wm.coords[:])
    bounds = shapely.geometry.polygon.Polygon(pia_wm_vertices)


    scale_transform = make_scale(working_scale)
    working_geo = geometries.transform(scale_transform)

    raster_stack = working_geo.rasterize()
    clear_overlaps(raster_stack)
    closest, closest_names = closest_from_stack(raster_stack)

    snapped_polys = get_snapped_polys(closest, closest_names)

    result_geos = Geometries()
    result_geos.register_polygons(snapped_polys)

    result_geos = (result_geos
        .transform(
            lambda ht, vt: (
                ht + working_geo.close_bounds.horigin,
                vt + working_geo.close_bounds.vorigin
            )
        )
        .transform(make_scale(1.0 / working_scale))
    )

    for key in list(result_geos.polygons.keys()):
        poly = result_geos.polygons[key].intersection(bounds)

        if isinstance(poly, shapely.geometry.GeometryCollection):
            poly = shapely.geometry.MultiPolygon([
                item for item in poly 
                if isinstance(item, shapely.geometry.Polygon)
            ])

        if isinstance(poly, shapely.geometry.MultiPolygon):
            max_area = -np.inf
            min_area = np.inf
            largest = None
            for item in poly:
                area = item.area
                if area > max_area:
                    max_area = area
                    largest = item
                if area < min_area:
                    min_area = area
            
            poly = largest

        result_geos.polygons[key] = poly

    boundaries = find_vertical_surfaces(
        result_geos.polygons, 
        layer_order, 
        pia=geometries.surfaces["pia"], 
        wm=geometries.surfaces["wm"]
    )

    result_geos.register_surfaces(boundaries)        

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
