from setuptools import setup, find_packages

setup(
    name='edgetpuvision',
    version='1.0',
    description='EdgeTPU camera API',
    long_description='API to run inference on image data coming from the camera.',
    author='Coral',
    author_email='coral-support@google.com',
    url="https://aiyprojects.withgoogle.com/",
    license='Apache 2',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.12.1',
        'Pillow>=4.0.0',
        'pygobject>=3.22.0',
        'protobuf>=3.0.0',
        'edgetpu',
    ],
    entry_points = {
        'console_scripts': ['edgetpu_classify=edgetpuvision.classify:main',
                            'edgetpu_classify_server=edgetpuvision.classify_server:main',
                            'edgetpu_detect=edgetpuvision.detect:main',
                            'edgetpu_detect_server=edgetpuvision.detect_server:main'],
    },
    python_requires='>=3.5.3',
)
