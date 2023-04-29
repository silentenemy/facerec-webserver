from distutils.core import setup

setup(
    name='facerec-webserver',
    version='0.1',
    packages=['facerec_webserver',],
	entry_points={
        'console_scripts': [
            'facerec-webserver=facerec_webserver.facerec_webserver:main',
        ]
    },
    license='MIT License',
    long_description=open('README.md').read(),
)
