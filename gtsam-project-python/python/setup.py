import os
import sys

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

packages = find_packages()

package_data = {
    package: [
        f
        for f in os.listdir(package.replace(".", os.path.sep))
        if os.path.splitext(f)[1] in (".so", ".pyd")
    ]
    for package in packages
}

dependency_list = open("/home/devicereal/github/gtsam-project-python/python/requirements.txt").read().split('\n')
dependencies = [x for x in dependency_list if x[0] != '#']

setup(
    name='gtsam_packing',
    description='Demo Library using GTSAM.',
    url='https://gtsam.org/',
    version='0.0.1',
    author="Varun Agrawal, Fan Jiang",
    author_email="varunagrawal@gatech.edu",
    license='Simplified BSD license',
    keywords="gtsam wrapper tutorial example",
    long_description=open("/home/devicereal/github/gtsam-project-python/README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    # https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    packages=packages,
    # Load the built shared object files
    package_data=package_data,
    include_package_data=True,
    test_suite="gtsam_example.tests",
    # Ensure that the compiled .so file is properly packaged
    zip_safe=False,
    platforms="any",
    install_requires=dependencies,
)
