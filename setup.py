import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "epdfpy",
    version = "0.0.1",
    author = "M.H.Kim, P.S.Kim and K-.Lee",
    author_email = "vader0210@skku.edu",
    description = "Python GUI module for pair distribution analysis on amorphous material",
    long_description = long_description,
    long_description_content_typew = "text/markdown",
    url = "https://github.com/GWlab-SKKU/ePDFpy",
    packages = setuptools.find_packages(),
    include_package_data=True,
    package_data={'epdfpy': ['assets/*','assets/css/*','assets/mask/*',
                             'assets/Parameter_files/*','settings/*']},

    # install_requires=[
    # ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'epdfpy = epdfpy.run_ui:main',
        ],
    },
    python_requires = '>=3.8'
)