find ./ -name "*.ipynb" -print | xargs -I {} jupyter nbconvert --clear-output --inplace {}
