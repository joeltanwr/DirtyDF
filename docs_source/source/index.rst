DirtyDF
=======
This page documents the DirtyDF Python package. If you have questions, please
email me at `vik.gopal@nus.edu.sg`.

Table of Contents
=================

.. toctree::
   :maxdepth: 2

   _autosummary/ddf_api_reference
   auto_examples/index
   samplers
   license

Introduction
============
This library is used for automatically adding random 'dirtyness' into existing datasets through the use of Stainers, to create multiple
variants of the original dataset, which are distinct from one another.

A common use case is as follows:

A tutor intends to give students an assignment/exam to test the students' data processing and analysis skills on a dataset.
In order to combat potential cheating, they would like to introduce random modifications to the single dataset to produce multiple
variants, so that the students' won't be able to easily cheat by copying each others' results or code.

The main components of this package is the DirtyDF data structure, and the various Stainers.

**DirtyDF** (DDF) is a wrapper around Pandas Dataframes that allow for ease-of-use with Stainers, which includes the history of stainers applied,
and for repeated runs to generate new, distinct, variants of the original DF.

**Stainer** is an object which acts on both standard Pandas Dataframes and DirtyDFs, in which its purpose is to apply the particular stainers'
'dirtyness' onto the dataset. Examples of stainers included in this package are Datetime-related stainers, Categorical-based stainers, Row Duplication stainers, etc.

The current implementation also allows for users to be able to create their own custom stainers for use with the DirtyDFs.

For more info on the existing stainers, check out the documentation on :doc:`Stainers<_autosummary/ddf.stainer>`.

For quick-start examples to get yourself familiar, check out our :doc:`Getting Started<auto_examples/index>` section.

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
