# USPECS: US Presidential Election Candidate Speeches Corpus

## Getting started
`data`folder is not included in the repo instead it is attached as release.

Update the `data`:
- clone repo and download release
- zip release and add the `data` folder along the repo.
- remove rows of unwanted speeches either in the `metadata.csv` or the `metadata.xlsx`. inside the `data`folder.
- run `preprocessing/dataUpdater.py` once. It will delete the corresponding txt files and update the graphic.
- zip the updated `data` folder and create a new release (increment version and document the changes)
