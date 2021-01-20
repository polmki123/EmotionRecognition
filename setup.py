import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepface",  
    version="0.0.48",
    author="Sefik Ilkin Serengil",
    author_email="serengil@gmail.com",
    description="A Lightweight Face Recognition and Facial Attribute Analysis Framework (Age, Gender, Emotion, Race) for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/serengil/deepface",
    packages=setuptools.find_packages(),
    scripts=['deepface/models/face-recognition-ensemble-model.txt'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='==3.6.5',
    install_requires=[  "numpy==1.19.3" ,"pandas==1.1.3", "tqdm>=4.30.0", "gdown>=3.10.1", 
                        "Pillow>=5.2.0", "opencv-python>=3.4.4", 
                        "keras-applications", "keras-preprocessing", "keras ", "tensorflow-gpu" ,"Flask>=1.1.2", "mtcnn>=0.1.0", "cython", "scipy==1.4.1", 
                        "astunparse", "h5py==2.10.0"]
)
