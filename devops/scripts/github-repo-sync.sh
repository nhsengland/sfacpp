#! /bin/bash
# -u the azure devops repo url
# -g the github repository url
# -n name of user for github
# -e email of user for github
# -b branch
# -m commit message
while getopts u:g:n:e:b:m: flag
do
    case "${flag}" in
        u) repo="${OPTARG}";;
        g) gh="${OPTARG}";;
        n) username="${OPTARG}";;
        e) email="${OPTARG}";;
        b) branch="${OPTARG}";;
        m) commit_msg="${OPTARG}";;
    esac
done
# check whether source repo is an existing directiry
if [ -d source_repo ]; then
    rm -rf source_repo;
fi
# create a source repository, and checkot main of the azure repo
git clone $repo source_repo
# check whether the destination repo exists, and delete otherwise
if [ -d gh_repo ]; then
    rm -rf gh_repo;
fi
# clone the destination github repository
git clone $gh gh_repo
# sync files from source to destination
echo "syncing files..."
rsync -cr source_repo/ gh_repo/ --exclude .git --delete
# commit and push changes to github
cd source_repo
git remote add origin $gh
git config user.name $username
git config user.email $email
git switch $branch || git switch -c $branch
git add .
git status
git commit -m "$commit_msg"
git push
