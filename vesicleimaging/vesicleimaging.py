"""Main module."""



# try:
#     from importlib.metadata import version, PackageNotFoundError
#
#     try:
#         __version__ = version(__name__)
#     except PackageNotFoundError:
#         pass
# except ImportError:
#     from pkg_resources import get_distribution, DistributionNotFound
#
#     try:
#         __version__ = get_distribution(__name__).version
#     except DistributionNotFound:
#         pass
#
#
# __all__ = [
#     "file_handling"
# ]

# import importlib
# import inspect
# import os
# import sys
#
# # Loop through all .py files in the package directory
# for module_file in os.listdir(os.path.dirname(__file__)):
#     if module_file.endswith('.py') and not module_file.startswith('__'):
#         # Remove the .py extension to get the module name
#         module_name = module_file[:-3]
#
#         # Import the module dynamically
#         module = importlib.import_module('.' + module_name, package=__name__)
#
#         # Get all the functions from the module and add them to the current namespace
#         for name, obj in inspect.getmembers(module):
#             if inspect.isfunction(obj):
#                 globals()[name] = obj
#
# # Clean up the namespace
# del importlib
# del inspect
# del os
# del sys
# del module_file
# del module_name
# del module