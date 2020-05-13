import os
import json
import xarray as xr

from typing import Dict, Any, Union, Optional

import boto3

import numpy as np
from neuron_morphology.swc_io import morphology_from_swc
import neuron_morphology.transforms.affine_transform as aff
from neuron_morphology.transforms.upright_angle.compute_angle import get_upright_angle, calculate_transform
from neuron_morphology.swc_io import morphology_to_swc
from neuron_morphology.morphology import Morphology

from harness import step_fns_ecs_harness


s3 = boto3.client("s3")


def collect_inputs(
        working_bucket: str, 
        run_prefix: str, 
        morphology_scaled_key: str,
        gradient_field_key: str
 ) -> Dict[str, Any]:
    """
    Gather from AWS the inputs required to run the depth field module.

    Parameters
    ----------
    working_bucket : The name of this pipeline's working bucket
    run_prefix : This run's resources (within the working bucket) have keys 
        beginning with this prefix.
    morphology_scaled_key : identifier for the scaled corrected morphology
    gradient_field_key: identifier for the gradient field

    Returns
    -------
    A dict with string keys:
        swc_file_str : swc file path
        gradient_field_file_str : gradient field file path
    """

    # boto3 get bytes from s3 working buckets
    gradient_field_obj = s3.get_object(Bucket=working_bucket, Key=gradient_field_key)
    gradient_field_data = xr.open_dataarray(gradient_field_obj["Body"].read())

    swc_file_obj = s3.get_object(Bucket=working_bucket, Key=morphology_scaled_key)
    morphology_data = morphology_from_swc(swc_file_obj["Body"])

    return morphology_data, gradient_field_data
    

def morphology_to_s3(bucket: str, key: str, dataset:Morphology):
    """
    Store a morphology object to s3 bucket as a swc file

    Parameters
    ----------
    bucket : name of the bucket in which to store this dataset
    key : at which to store the dataset
    dataset : The dataset to store

    Returns
    -------
    key : the argued key
    """

    tmp_path = key.split("/")[-1]
    morphology_to_swc(dataset, tmp_path)
    s3.upload_file(Filename=tmp_path, Bucket=bucket, Key=key)

    os.remove(tmp_path)
    return key


def put_outputs(
        bucket: str, 
        prefix: str, 
        upright_transform: Dict[str, Any], 
        upright_angle: str, 
        upright_morphology: Morphology
) -> Dict[str, Any]:
    """Write this module's outputs to s3 & prepare its step functions 
    response.

    Parameters
    ----------
    bucket : the name of the bucket to which the upright transform outputs will be written
    prefix : keys at which the upright transform outputs are stored will begin with this prefix
    upright_transform : the dict of the upright transform matrix
    upright_angle : the angle of the upright direction 
    upright_morphology : the upright transformed morphology. 

    Returns
    -------
    Outputs to send back to step functions. These are:
        upright_transform_dict_key : s3 key of the upright transform matrix
        upright_angle_key : s3 key of the upright angle
        upright_morphology_key : s3 key of the upright transformed morphology 
    """

    return {
        'upright_transform_dict_key': upright_transform,
        'upright_angle_key': upright_angle,
        "upright_morphology_key": morphology_to_s3(
            bucket, f"{prefix}/upright_morphology.swc", upright_morphology
        )

    }


def run_upright_transform(token: Optional[str] = None):
    """Entry point for running the upright transform step from a 
    step-functions-managed ECS instance
    """

    working_bucket = os.environ["WORKING_BUCKET"]
    run_prefix = os.environ["RUN_PREFIX"]
    gradient_field_key = os.environ["GRADIENT_FIELD_KEY"]
    morphology_scaled_key = os.environ["MORPHOLOGY_SCALED_KEY"]
 
    morphology, gradient_field = collect_inputs(working_bucket, run_prefix, morphology_scaled_key, gradient_field_key)

    # find the upright direction at the soma location
    outputs = calculate_transform(gradient_field, morphology)

    # apply affine transform
    upright_morphology = outputs["upright_transform"].transform_morphology(morphology, clone=True)

    return put_outputs(
        working_bucket,
        run_prefix,
        outputs["upright_transform"].to_dict(),
        outputs["upright_angle"],
        upright_morphology
    )


def main():
    step_fns_ecs_harness(run_upright_transform)


if __name__ == "__main__":
    main()