# edgetpuvision

Python API to run inference on image data coming from the camera.

## Build

python3 setup.py sdist
python3 setup.py bdist
python3 setup.py sdist_wheel

## Debian pacakge

Install `stdeb` package by running `apt-get install python3-stdeb` or
`pip3 install stdeb`. Then to generate debian folder run:
```
python3 setup.py --command-packages=stdeb.command debianize
```

To build debian pacakge run:
```
dpkg-buildpackage -b -rfakeroot -us -uc -tc
```
