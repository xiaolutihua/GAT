from binbin.application import get_random_app, ApplicationDAG
from binbin.devices import IoT, Fog, Cloud
from binbin.env import BinbinSimEnv
from common.data import CPU, Memory
from constants import G


def make_iot_s(env: "BinbinSimEnv", app: "ApplicationDAG" = None) -> "IoT":
    if app is None:
        iot = IoT(env, get_random_app())
    else:
        iot = IoT(env, app)
    iot.computing_units = CPU(env, 1)
    iot.compute_speed = 0.7 * G
    iot.memory = Memory(env, 0.25 * G)
    return iot


def make_iot_m(env: "BinbinSimEnv") -> "IoT":
    iot = IoT(env, get_random_app())
    iot.computing_units = CPU(env, 2)
    iot.compute_speed = 0.9 * G
    iot.memory = Memory(env, 0.5 * G)
    return iot


def make_iot_l(env: "BinbinSimEnv") -> "IoT":
    iot = IoT(env, get_random_app())
    iot.computing_units = CPU(env, 2)
    iot.compute_speed = 1.2 * G
    iot.memory = Memory(env, 1 * G)
    return iot


def make_iot_x(env: "BinbinSimEnv") -> "IoT":
    iot = IoT(env, get_random_app())
    iot.computing_units = CPU(env, 4)
    iot.compute_speed = 1.5 * G
    iot.memory = Memory(env, 2 * G)
    return iot


def make_fog_s(env: "BinbinSimEnv") -> "Fog":
    fog = Fog(env)
    fog.computing_units = CPU(env, 8)
    fog.compute_speed = 2.9 * G
    fog.memory = Memory(env, 16 * G)
    fog.price_factor = 3.0e-4
    return fog


def make_fog_m(env: "BinbinSimEnv") -> "Fog":
    fog = Fog(env)
    fog.computing_units = CPU(env, 8)
    fog.compute_speed = 3.2 * G
    fog.memory = Memory(env, 16 * G)
    fog.price_factor = 4.2e-4
    return fog


def make_fog_l(env: "BinbinSimEnv") -> "Fog":
    fog = Fog(env)
    fog.computing_units = CPU(env, 12)
    fog.compute_speed = 3.7 * G
    fog.memory = Memory(env, 32 * G)
    fog.price_factor = 5.4e-4
    return fog


def make_fog_x(env: "BinbinSimEnv") -> "Fog":
    fog = Fog(env)
    fog.computing_units = CPU(env, 16)
    fog.compute_speed = 4.2 * G
    fog.memory = Memory(env, 32 * G)
    fog.price_factor = 6.6e-4
    return fog


def make_cloud_m(env: "BinbinSimEnv") -> "Cloud":
    cloud = Cloud(env)
    cloud.computing_units = CPU(env, 32)
    cloud.compute_speed = 3.2 * G
    cloud.memory = Memory(env, 32 * G)
    cloud.price_factor = 1.21e-3
    return cloud


def make_cloud_l(env: "BinbinSimEnv") -> "Cloud":
    cloud = Cloud(env)
    cloud.computing_units = CPU(env, 64)
    cloud.compute_speed = 3.7 * G
    cloud.memory = Memory(env, 32 * G)
    cloud.price_factor = 1.97e-3
    return cloud
