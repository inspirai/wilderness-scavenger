import os
import ctypes
from ctypes import cdll
from sys import platform

import trimesh
import numpy as np
from math import radians
from numpy.ctypeslib import ndpointer
from typing import List


def perspective_frustum(hw_ratio, x_fov, znear, zfar):
    assert znear != zfar
    right = np.abs(np.tan(x_fov) * znear)
    top = right * hw_ratio
    left = -right
    bottom = -top
    return [left, right, bottom, top, znear, zfar]


class RaycastManager(object):
    DEFAULT_HEIGHT = 22
    DEFAULT_WIDTH = 38
    DEFAULT_FAR = 100
    __VISION_ANGLE = 60

    def __init__(self, mesh_file_path):
        self.mesh_file_path = mesh_file_path
        self.HEIGHT = self.DEFAULT_HEIGHT
        self.WIDTH = self.DEFAULT_WIDTH
        self.FAR = self.DEFAULT_FAR

        if platform.startswith("linux"):
            lib_filename = "libraycaster.so"
        elif platform.startswith("darwin"):
            lib_filename = "libraycaster.dylib"
        elif platform.startswith("win"):
            lib_filename = "raycaster.dll"
        else:
            raise NotImplementedError(f"{platform} is not supported")

        work_dir = os.path.dirname(__file__)
        lib_path = os.path.join(work_dir, "lib", lib_filename)
        self.ray_lib = cdll.LoadLibrary(lib_path)

        try:
            c_func = self.ray_lib.init_mesh
            c_func.argtypes = [
                ctypes.c_void_p,  # ray_tracer_ptr
                ndpointer(
                    ctypes.c_float, flags="C_CONTIGUOUS"
                ),  # vertices. (num_vertices, 3)
                ctypes.c_size_t,  # num_vertices
                ndpointer(
                    ctypes.c_uint32, flags="C_CONTIGUOUS"
                ),  # faces. (num_faces, 3)
                ctypes.c_size_t,  # num_faces
            ]
            c_func.restype = ctypes.c_void_p

            c_func = self.ray_lib.get_depth
            c_func.argtypes = [
                ctypes.c_void_p,
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ]

            c_func = self.ray_lib.free_arrays
            c_func.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
            ]

            c_func = self.ray_lib.agent_is_visible
            c_func.argtypes = [
                ctypes.c_void_p,  # ray_tracer_ptr
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # body param
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # view param
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # position
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # cameralocation
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),  # cameralocation
                ctypes.c_size_t,  # num_vertices
                ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"),  # team_id
                ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"),  # is_visiable info
            ]
            c_func.restype = ctypes.c_void_p

            c_func = self.ray_lib.free_mesh
            c_func.argtypes = [ctypes.c_void_p]

        except Exception:
            print("External library not loaded correctly: {}".format(lib_filename))

        self.depth_ptr = ctypes.POINTER(ctypes.c_void_p)()

        mesh = trimesh.load(mesh_file_path, force="mesh")
        v = np.array(mesh.vertices).astype(np.float32)
        f = np.array(mesh.faces).astype(np.uint32)

        c_func = self.ray_lib.init_mesh
        self.depth_ptr = c_func(self.depth_ptr, v, int(v.shape[0]), f, int(f.shape[0]))

    def get_depth(
        self,
        position: List[float],
        direction: List[float],
    ):
        """
        multi agent support todo@wsp
        position: position of only 1 agent for now
        ray_origin and ray_direciton: 2d list
        """

        height = self.HEIGHT
        width = self.WIDTH
        far = self.FAR

        x, y, z = position
        position_in_mesh = np.asarray([-x, y, z])  # negative x
        r = np.asarray(direction) * np.pi / 180
        cam_lookat = position_in_mesh + np.asarray(
            [-np.sin(r[2]) * np.cos(r[1]), -np.sin(r[1]), np.cos(r[2]) * np.cos(r[1])]
        )  # negative x

        num_cameras = 1
        out_depth_values_ptr = (ctypes.POINTER(ctypes.c_float) * num_cameras)()
        cam_param_array_double = np.zeros(
            (num_cameras, 18), dtype=np.float64, order="C"
        )
        for i in range(num_cameras):
            cam_pos = np.array(position_in_mesh[i * 3 : i * 3 + 3])
            cam_param_array_double[i, 0:3] = cam_pos
            cam_param_array_double[i, 3:6] = cam_lookat
            cam_param_array_double[i, 6:9] = [0, 1, 0]
            cam_param_array_double[i, 9:10] = 1.0
            cam_param_array_double[i, 10:16] = perspective_frustum(
                hw_ratio=float(height) / width,
                x_fov=radians(self.__VISION_ANGLE // 2),
                znear=0.01,
                zfar=far,
            )
            cam_param_array_double[i, 16:18] = [height, width]

        c_func = self.ray_lib.get_depth
        c_func(
            self.depth_ptr,
            cam_param_array_double,
            int(cam_param_array_double.shape[0]),
            out_depth_values_ptr,
        )

        out_depth_maps = []
        for i in range(num_cameras):
            depth_map = np.ctypeslib.as_array(
                out_depth_values_ptr[i], shape=(height, width)
            ).copy()
            depth_map[np.isnan(depth_map)] = far
            out_depth_maps.append(depth_map)

        self._free(out_depth_values_ptr)

        return out_depth_maps

    def agent_is_visible(
        self,
        body_param,
        view_angle,
        agent_team_id,
        positon,
        cameralocation,
        camerarotation,
    ):
        agent_num = len(agent_team_id)
        c_func = self.ray_lib.agent_is_visible
        is_visiable = np.zeros((agent_num, agent_num), dtype=np.uint32, order="C")

        c_func(
            self.depth_ptr,
            np.array(body_param, dtype=np.float32, order="C"),
            np.array(view_angle, dtype=np.float32, order="C"),
            np.array(positon, dtype=np.float32, order="C"),
            np.array(cameralocation, dtype=np.float32, order="C"),
            np.array(camerarotation, dtype=np.float32, order="C"),
            int(agent_num),
            np.array(agent_team_id, dtype=np.uint32, order="C"),
            is_visiable,
        )
        return is_visiable

    def _free(self, ptr):
        num_arrays = len(ptr)
        arr_ptr_void = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))
        c_func = self.ray_lib.free_arrays
        c_func(arr_ptr_void, num_arrays)

    def _free_mesh(self):
        c_func = self.ray_lib.free_mesh
        c_func(self.depth_ptr)

    def __repr__(self) -> str:
        return f"RayTracer(HEIGHT={self.HEIGHT}, WIDTH={self.WIDTH}, DEPTH={self.FAR}, mesh={self.mesh_file_path})"

    def __del__(self):
        self._free_mesh()
        print(f"[free] memory used by {self.mesh_file_path} is freed")
