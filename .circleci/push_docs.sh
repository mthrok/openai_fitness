#!/bin/bash
set -eu

if [[ ( "${CIRCLE_PROJECT_USERNAME:-false}" = "mthrok" ) && ( "$(git rev-parse --abbrev-ref HEAD)" = "master" ) ]]; then
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
