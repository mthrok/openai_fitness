#!/bin/bash
set -eu

rm -rf docs/source/API/*.rst
sphinx-apidoc -e -o docs/source/API luchador --force
make -C docs html

if [[ ${CIRCLECI:=false} = true ]]; then
    git config user.email "mthrok@gmail.com"
    git config user.name "CCI"
    COMMIT="$( git log --pretty=oneline -1 )"

    # Clean up the current repository so that we can change to gh-pages branch
    git reset --hard

    # Update remote branch
    git fetch
    git checkout gh-pages
    git reset --hard origin/gh-pages

    # Update html dir
    rm -rf html
    mv docs/build/html ./
    git add ./html

    # Update remote branch
    git commit --allow-empty -m "[skip ci] ${COMMIT}"
    git push
fi
