import ir_datasets
dataset = ir_datasets.load('msmarco-passage/train')
# Documents
for doc in dataset.docs_iter():
    print(doc)
# GenericDoc(doc_id='0', text='The presence of communication amid scientific minds was equa...
# GenericDoc(doc_id='1', text='The Manhattan Project and its atomic bomb helped bring an en...
# ...