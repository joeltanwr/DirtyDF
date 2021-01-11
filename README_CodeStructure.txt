Explanation of Structure

example.ipynb: This is just a simple example to see how something can currently work

artificialdata.py: 
- DirtyDF is where the DF and history information will be stored
- HistoryDF is to facilitate the above class
- Combiner acts as the glue for all the stainers. Calling transform_all is what will activate all the stainers

stainers.py:
- Stainer is just an interface of how all stainers should minimally look like
- AddDuplicate is a specific stainer that adds duplicates to a dataset

errors.py:
- Created it to help us eventually debug and control the flow of the whole combiner process 
- There will likely be errors due to user input / some obscure cases and we won't want to kill the whole process once we hit any of these errors so the classes can help to segregate our errors and determine what are the ones that needs to be handled
