## Add SuperGlue as Git Submodule

Add SuperGlue to the project's folder:

```bash
git submodule add https://github.com/magicleap/SuperGluePretrainedNetwork superglue
```

**Note:**
<br>
Installing dependencies for SuperGlue fails with error: 
`No matching distribution found for opencv-python==4.1.2.30`

**Solution:**
<br>
Update file `requirements.txt` to use a newer version of OpenCV:
`opencv-python>=4.1.2.30`

Install python requirements for SuperGlue:

```bash
pip install -r SuperGluePretrainedNetwork/requirements.txt
```