from datasets import load_dataset

ds = load_dataset("ds4sd/FinTabNet_OTSL")

print('Dataset is')
print(ds)

print('Sample item value in dataset')
print(ds['train'][0])






