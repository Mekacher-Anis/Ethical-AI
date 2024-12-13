### Annotation Tool setup

1. Convert dataset
   1. Run `convert_dataset_to_jsonl.py` to make it compatible with the annotation tool (in word_classification, assumes that datasets are in datasets/appropriateness-corpus)
2. Install and setup doccano
   1. Run: `pip install doccano`
   * maybe you have to downgrade the numpy version to 1.26 manually
   1. Run `doccano createuser --username user --password password`
   2. Run `doccano webserver`
   3. Run in another terminal: `doccano task`
   4. Go to `http://127.0.0.1:8000/` in your browser
3. Create doccano project
   1. Login with your credentials
   2. Create a sequence labeling project
   3. Add a project name and description
   4. Select the option: `Allow overlapping spans`
   5. Create project
4. Setup doccano project
   1. Import labels from `word_classification/doccano/label_config.json`
   2. Add guideline text from `word_classification/doccano/guideline.md`
   3. Correspond to team members which data samples have to be annotated and edit the `word_classification/doccano/test.jsonl` file accordingly
   4. Upload the dataset
      1. Select JSONL as file format
      2. Column data needs to be `post_text`
      3. Column label can be ignored for now
      4. Upload your test corpus json file
      5. Click import
5. Start annotating
   1. Select all characters you want to label and mark them accordingly
   2. Note: You can mark the text sample as done/checked by clicking on the cross
   3. Note: You can see the document level annotations in the bottom right (only third level for brevity)
6. Export the new annotated dataset
7. TODO make exported dataset compatible with downstream tasks