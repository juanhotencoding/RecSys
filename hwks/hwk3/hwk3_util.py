# Homework 3
# INFO 4871/5871
# Professor Robin Burke

from lenskit import util


# The whole cloning system in LKPy is a still inflexible, since it assumes a very particular object structure.
# The purpose of this function is to add the capability for an object to have its own `clone` function, which will
# then override the built-in one if that is appropriate.
def my_clone(obj):
    if hasattr(obj, 'clone'):
        return obj.clone()
    else:
        return util.clone(obj)

