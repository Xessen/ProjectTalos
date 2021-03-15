from setuptools import setup,find_packages

setup(
    name="projecttalos", # 
    version="0.0.3",
    author="Şeyhmus Baskın",
    author_email="seyhmusbaskin@yandex.com",
    description="A small Machine Learning package",
    url="https://github.com/Xessen/ProjectTalos",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.3",
    install_requires=["numpy>=1.15.0"]
    )