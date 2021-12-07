 #!/usr/bin/env python

from setuptools import find_packages, setup


def setup_package():
    setup(
        name="CRVAE",
        version="0.1",
        description="CRVAE",
        packages=find_packages(),
    )


if __name__ == "__main__":
    setup_package()
