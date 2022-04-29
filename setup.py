from setuptools import setup
import pathlib


here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# non-python data that needs to be installed along with the main python modules
package_data = ["__init__.py", "lib/*"]


setup(
    name="inspirai_fps",
    version="0.0.3",
    author="Inspir.AI",
    author_email="cog2022@inspirai.com",
    url="https://github.com/inspirai/wilderness-scavenger",
    description="An intelligent agent learning platform based on a 3D open-world FPS game",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["inspirai_fps"],
    python_requires=">=3.8, <4",
    package_data={"inspirai_fps": package_data},
    install_requires=["Pillow", "numpy", "grpcio", "rich", "protobuf", "trimesh"],
    extras_require={"baseline": ["gym", "ray[rllib]", "torch"]},
    keywords=[
        "inspirai",
        "fps",
        "game",
        "open world",
        "ai",
        "deep learning",
        "reinforcement learning",
        "research",
    ],
    license="LICENSE",
)
