import io
import os
import time
import logging
import tarfile
from typing import Any
from pathlib import Path
from datetime import datetime

import docker

from .task_run import TaskRun
from .docker_image_utils import build_image


def remove_container(client: docker.DockerClient, name: str, logger: logging.Logger):
    try:
        container = client.containers.get(name)
        if container:
            container.reload()
            container.stop()
            container.remove(force=True)
            time = datetime.now()
            logger.debug(f"Removed existing container: {name}, {time}")
    except docker.errors.NotFound:
        pass


def copy_to_container(container, src: Path, dest: Path, unpack=False):
    """
    Copies files from the host to the container.

    Args:
        container (docker.models.containers.Container): The target container.
        src (Path or str): Source path on the host.
        dest (Path or str): Destination path inside the container.
        unpack (bool): If True, unpacks the content of the source folder into the destination.
                       If False, copies the source folder itself into the destination.
    """

    if not src.exists():
        raise FileNotFoundError(f"Source path '{src}' does not exist.")

    # Ensure destination directory exists inside the container
    mkdir_cmd = ["mkdir", "-p", str(dest)]
    exit_code, output = container.exec_run(cmd=mkdir_cmd)
    if exit_code != 0:
        raise RuntimeError(f"Failed to create directory '{dest}' inside the container.")

    # Create a tar archive of the source directory
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        for root, _dirs, files in os.walk(src):
            for file in files:
                file_path = Path(root) / file
                if unpack:
                    # If unpacking, we need to adjust the arcname to be relative to the content of src
                    arcname = file_path.relative_to(src)
                else:
                    # Otherwise, preserve the folder structure, so arcname includes src folder
                    arcname = file_path.relative_to(src.parent)
                tar.add(file_path, arcname=arcname)
    tar_stream.seek(0)

    # Upload the tar archive to the container
    success = container.put_archive(path=str(dest), data=tar_stream)
    if not success:
        raise RuntimeError(f"Failed to copy files to container '{container.name}'.")


def copy_from_container(container, src: Path, dest: Path, tag: str = ""):
    """
    Copies a file or directory from the Docker container to the host machine.
    :param container: Docker container object
    :param src: Absolute path to the file or directory inside the container
    :param dest: Absolute path on the host machine where the file or directory will be saved
    """
    try:
        # Retrieve the file or directory as a tar archive
        bits, stat = container.get_archive(src)

        # Combine the tar chunks into a single bytes object
        file_like_object = io.BytesIO()
        for chunk in bits:
            file_like_object.write(chunk)
        file_like_object.seek(0)

        # Ensure the host directory exists
        os.makedirs(dest, exist_ok=True)

        # Extract the contents from the tar archive
        with tarfile.open(fileobj=file_like_object) as tar:
            tar.extractall(path=dest)

        print(f"{tag} Successfully copied '{src}' to '{dest}'.")
    except Exception as e:
        print(f"{tag} Failed to copy from container: {e}")


def run_in_container(
    client: docker.DockerClient,
    task_run: TaskRun,
    logger: logging.Logger,
    environment: dict,
    retain_container: bool = True,
) -> dict[str, Any]:
    start_time = time.time()

    image_name = task_run.image if task_run.image else "default_evals_image"
    container_name = f"{task_run.task_id}_container"
    run_assets_path = task_run.run_asset_path
    # Localhost path where to save the output
    output_path = task_run.run_dir / task_run.run_id / task_run.task_id

    try:
        build_image(client=client, image_name=image_name, build_dir=output_path, force_build=False)
    except Exception as e:
        logger.error(f"Failed to build Docker image '{image_name}': {e}")
        return
    try:
        remove_container(client=client, name=container_name, logger=logger)
        environment = {
            **environment,
            "TASK_ID": task_run.task_id,
            "INSTRUCTION": task_run.instruction,
            "OUTPUT_DIR": "/home/output",  # Define OUTPUT_DIR as an environment variable
        }
        container = client.containers.create(
            image=image_name,
            name=container_name,
            environment=environment,
            tty=True,
            detach=True,
        )
        logger.debug("container created")
        container.start()
        logger.debug("container started")
        container_started_time = time.time()
        setup_time = round(container_started_time - start_time, 2)

        # Once container starts, copy assets to /home
        copy_to_container(
            container=container,
            src=run_assets_path,
            dest=Path("/home"),
            unpack=True,
        )

        command = [
            "/bin/bash",
            "-c",
            "/home/entrypoint.sh",
        ]

        exit_code, output = container.exec_run(command, stream=True, user="nonroot")
        for chunk in output:
            logger.info(f"[Container] {chunk.decode('utf-8').strip()}")
        end_time = time.time()
        copy_from_container(
            container=container,
            src="/home/output",
            dest=output_path,
        )
        return {
            "status": "success",
            "exit_code": exit_code,
            "container_setup_time": setup_time,
            "duration": round(end_time - container_started_time, 2),
        }
    except docker.errors.APIError as api_err:
        raise api_err

    except Exception as e:
        raise e
    finally:
        if not retain_container:
            remove_container(client=client, name=container_name, logger=logger)
