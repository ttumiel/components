# Models

For better reusability, all models inherit from `ModelBuilder` in `base.py`. This class allows simple composition and manipulation of a sequence of blocks. The constructor takes the block class or a list of blocks as the first parameter and builds a sequential model using the arguments provided in the second parameter (as a list). Additionally, a head and a stem can be added.
