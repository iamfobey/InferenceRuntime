from conan import ConanFile
from conan.tools.cmake import CMakeDeps


class InferenceRuntimeConan(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    requires = (
        "simdjson/4.6.1",
        "gtest/1.17.0",
        "benchmark/1.9.5",
        "spdlog/1.17.0",
    )
    default_options = {"spdlog/*:use_std_fmt": False}

    def generate(self):
        CMakeDeps(self).generate()
