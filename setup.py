from setuptools import setup

setup_requires = []

install_requires = ["dataclasses", "numpy", "Pillow"]

setup(
    name="inpaint",
    version="0.0.1",
    description="Just a wrapper",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    install_requires=install_requires,
)
