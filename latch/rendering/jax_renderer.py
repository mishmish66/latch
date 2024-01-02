# Before anything else is imported, set the environment variable to use the right rendering backend
import os

os.environ["MUJOCO_GL"] = "EGL"

import mujoco
from mujoco import mjx

import jax
from jax import numpy as jnp
from jax.tree_util import Partial, register_pytree_node_class

import numpy as np

from multiprocessing import Process, Event, Queue
from queue import Empty as QueueEmptyException

from typing import Optional, Union


@register_pytree_node_class
class JAXRenderer:
    def __init__(
        self,
        model: mujoco.MjModel,  # type: ignore
        height: int = 240,
        width: int = 320,
        max_geom: int = 10000,
        num_workers: Optional[int] = None,
    ):
        if num_workers is None:
            # Set the number of workers to the number of threads
            # Temporarily set the number of threads to 1 for debugging
            # num_workers = os.cpu_count()
            num_workers = 1
        if num_workers is None:
            # Failed to get the number of threads, set to 1
            num_workers = 1

        self._num_workers = num_workers
        self._workers = []
        self._queue = Queue()

        self._model = model
        self._width = width
        self._height = height
        self._max_geom = max_geom

        self._init_workers()

    def _init_workers(self):
        for _ in range(self._num_workers):
            renderer = mujoco.Renderer(
                self._model, self._width, self._height, self._max_geom
            )
            self._workers.append(JAXRenderWorker(renderer, self._queue))

    def _host_render(
        self,
        data: mjx.Data,
        camera: Union[int, str] = -1,
    ):
        host_data = mjx.get_data(self._model, data)
        is_batch = isinstance(host_data, list)
        if is_batch:
            tasks = []
            host_datas = host_data
            for host_data in host_datas:
                task = JAXRenderWorkerTask(host_data, camera)
                self._queue.put(task)
                tasks.append(task)
            imgs = []
            for task in tasks:
                imgs.append(task.await_result())
            return np.stack(imgs, axis=0)

        else:
            task = JAXRenderWorkerTask(host_data, camera)
            self._queue.put(task)
            img = task.await_result()
            return img

    @Partial(jax.jit, static_argnames=("camera",))
    def render(
        self,
        data: mjx.Data,
        camera: Union[int, str] = -1,
    ):
        # Define a callback function to bind camera to the function
        # This avoids it being interpreted as a JAX type in case it is a string
        def camera_bound_render_callback(data):
            return self._host_render(data, camera)

        img: jax.Array = jax.pure_callback(  # type: ignore
            callback=camera_bound_render_callback,
            result_shape_dtypes=jnp.zeros(
                [3, self._width, self._height],
            ),
            data=data,
        )

        return img

    def close(self):
        for worker in self._workers:
            worker.close()

        self._queue.join()

    def tree_flatten(self):
        return (), {
            "num_workers": self._num_workers,
            "workers": self._workers,
            "queue": self._queue,
            "model": self._model,
            "width": self._width,
            "height": self._height,
            "max_geom": self._max_geom,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Rebuid the renderer with the same pieces
        renderer = cls.__new__(cls)

        renderer._num_workers = aux_data["num_workers"]
        renderer._workers = aux_data["workers"]
        renderer._queue = aux_data["queue"]
        renderer._model = aux_data["model"]
        renderer._width = aux_data["width"]
        renderer._height = aux_data["height"]
        renderer._max_geom = aux_data["max_geom"]

        return renderer


class JAXRenderWorkerTask:
    def __init__(
        self,
        data,
        camera: Union[int, str] = -1,
    ):
        self._result_img: Optional[np.ndarray] = None
        self.fulfill_event = Event()
        self.data = data
        self.camera = camera

    def await_result(self):
        self.fulfill_event.wait()
        return self._result_img

    def fulfill(self, img: np.ndarray):
        self._result_img = img
        self.fulfill_event.set()


class JAXRenderWorker:
    def __init__(
        self,
        renderer: mujoco.Renderer,
        queue: Queue,
    ):
        self._renderer = renderer

        self._running = True
        # Launch the worker process
        self._process = Process(target=self._run, args=(queue,))
        self._process.start()

    def _run(self, queue):
        while self._running:
            try:
                task = queue.get(timeout=1)
                img = self._render(task)
                task.fulfill(img)
            except QueueEmptyException:
                # No task, check if we should stop then continue
                pass

    def _render(self, task: JAXRenderWorkerTask):
        host_data = mjx.get_data(self._renderer.model, task.data)
        self._renderer.update_scene(host_data, task.camera)
        img = self._renderer.render()

        return img

    def close(self):
        self._running = False
        self._process.join()
        self._renderer.close()
