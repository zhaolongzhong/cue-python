#!/usr/bin/env bash

set -euo pipefail

# Step 1: Get version name from branch name
branch_name=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $branch_name"

# Extract platform from the branch name
platform=$(echo $branch_name | cut -d '/' -f 2)
echo "Platform: $platform"

# Validate platform
if [ -z "$platform" ]; then
    echo "Error: Could not determine platform from branch name"
    exit 1
fi

# Extract version name
version_name=${branch_name##*/}
echo "Version name: $version_name"

# Validate version name format
if [[ ! $version_name =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Invalid version name format. Expected 'x.x.x', got '$version_name'"
    exit 1
fi

# Step 2: Get version code from latest tag
git fetch --depth=1 origin +refs/tags/*:refs/tags/*
latest_tag=$(git tag --sort=-v:refname | grep "^${platform}-" | head -1)
echo "Latest tag: $latest_tag"

# Extract and increment version code
version_code="${latest_tag##*-}"
version_code=$((version_code + 1))

# Prepare new tag
new_tag="$platform-$version_name-$version_code"
echo "New tag: $new_tag"
echo "New version code: $version_code"
echo "New version name: $version_name"

# Validate tag format and increment
if [[ ! $new_tag =~ ^$platform-[0-9]+\.[0-9]+\.[0-9]+-[0-9]+$ ]]; then
    echo "Invalid new tag format. Expected '$platform-x.x.x-x', got '$new_tag'"
    exit 1
fi

latest_version_code=${latest_tag##*-}
new_version_code=${new_tag##*-}
if [[ $((new_version_code - latest_version_code)) -ne 1 ]]; then
    echo "Invalid version code increment. Expected '$latest_version_code' -> '$new_version_code'"
    exit 1
fi

# Output to GitHub environment
echo "new_tag=${new_tag}" >> $GITHUB_ENV
echo "version_code=${version_code}" >> $GITHUB_ENV
echo "version_name=${version_name}" >> $GITHUB_ENV
