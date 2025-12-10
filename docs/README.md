# AMS Documentation

##  Build Instructions

To get setup with a virtual environment run:

```bash
python3 -m venv -m ams-doc
. ams-doc/bin/activate
cd AMS/docs
pip install -r requirements.txt
```

Then you can build the documention locally with:
```bash
make html
```

This step can take > 2 minutes due to the use of Exhale which reads all the Doxygen XML files
and generates a nice Sphinx-like documentation.