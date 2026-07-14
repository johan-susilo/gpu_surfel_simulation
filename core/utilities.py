import functools
import warnings
import os
from pathlib import Path
import re


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def cleanFolders(dummyRun=False):
    ''''deleting files in folders with certain file patterns, but always keeping the latest ix while removing the rest'''
    
    rootfolder = Path('output')
    folderSelect = list(rootfolder.glob('**/**/simLog'))
    filePatterns = ['simLog_*.vtp', 'simLog_*.pkl']
    for simFolder in folderSelect:
        for filePattern in filePatterns:
            pattern = filePattern.replace('*', '(\\d*)')  # check if this is still correct
            filelistSort = [(file, int(re.findall(pattern, str(file))[0])) for file in list(simFolder.glob(filePattern))]
            filelistSort = sorted(filelistSort, key=lambda tup: tup[1], reverse=True)
            for ix, (file, ixFile) in enumerate(filelistSort):
                if ix==0:
                    print(f'<dummyrun={dummyRun}> : Keeping file {file}')
                    continue
                print(f'<dummyrun={dummyRun}> : Removing file {file}')
                if not dummyRun:
                    os.remove(file)
